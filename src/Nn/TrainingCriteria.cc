/*
 * Copyright 2016 Alexander Richard
 *
 * This file is part of Squirrel.
 *
 * Licensed under the Academic Free License 3.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should have received a copy of the License along with Squirrel.
 * If not, see <https://opensource.org/licenses/AFL-3.0>.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

/*
 * TrainingCriteria.cc
 *
 *  Created on: Jun 5, 2014
 *      Author: richard
 */

#include "TrainingCriteria.hh"
#include "ActivationLayer.hh"

using namespace Nn;

/*
 * TrainingCriterion
 */
const Core::ParameterEnum TrainingCriterion::paramTrainingCriterion_("training-criterion",
		"none, dummy, cross-entropy, weighted-cross-entropy, squared-error, weighted-squared-error, smoothed-l1, multi-task-training", "none", "");

void TrainingCriterion::computeInitialErrorSignal(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	initialErrorSignal(network.outputLayer().latestActivations(0), targets, network.outputLayer().latestErrorSignal(0));
}

void TrainingCriterion::computeInitialErrorSignals(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	require_eq(network.outputLayer().nTimeframes(), targets.nTimeframes());
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		initialErrorSignal(network.outputLayer().activations(t, 0), targets.at(t), network.outputLayer().errorSignal(t, 0));
	}
}

Float TrainingCriterion::computeObjectiveFunction(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	return objectiveFunction(network.outputLayer().latestActivations(0), targets);
}

Float TrainingCriterion::computeObjectiveFunction(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	require_eq(network.outputLayer().nTimeframes(), targets.nTimeframes());
	Float objFctn = 0;
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		objFctn += objectiveFunction(network.outputLayer().activations(t, 0), targets.at(t));
	}
	return objFctn;
}

u32 TrainingCriterion::nClassificationErrors(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	require_eq(network.outputLayer().latestActivations(0).nRows(), targets.nRows());
	require_eq(network.outputLayer().latestActivations(0).nColumns(), targets.nColumns());
	return network.outputLayer().latestActivations(0).nClassificationErrors(targets);
}

u32 TrainingCriterion::nClassificationErrors(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	require_eq(network.outputLayer().nTimeframes(), targets.nTimeframes());
	u32 errors = 0;
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		require_eq(network.outputLayer().activations(t, 0).nRows(), targets.at(t).nRows());
		require_eq(network.outputLayer().activations(t, 0).nColumns(), targets.at(t).nColumns());
		errors += network.outputLayer().activations(t, 0).nClassificationErrors(targets.at(t));
	}
	return errors;
}

TrainingCriterion* TrainingCriterion::createCriterion() {
	TrainingCriterion* res = 0;
	switch ((CriterionType) Core::Configuration::config(paramTrainingCriterion_)) {
		case dummy:
			res = new DummyCriterion();
			Core::Log::os("Use dummy criterion.");
			break;
		case crossEntropy:
			res = new CrossEntropyCriterion();
			Core::Log::os("Use cross-entropy criterion.");
			break;
		case weightedCrossEntropy:
			res = new WeightedCrossEntropyCriterion();
			Core::Log::os("Use weighted-cross-entropy criterion.");
			break;
		case squaredError:
			res = new SquaredErrorCriterion();
			Core::Log::os("Use squared-error criterion.");
			break;
		case weightedSquaredError:
			res = new WeightedSquaredErrorCriterion();
			Core::Log::os("Use weighted-squared-error criterion.");
			break;
		case smoothedL1:
			res = new SmoothedL1Criterion();
			Core::Log::os("Use smoothed-l1 criterion.");
			break;
		case multiTaskTraining:
			res = new MultiTaskTrainingCriterion();
			Core::Log::os("Use multi-task-training criterion.");
			break;
		default:
			Core::Error::msg("TrainingCriterion.cc: no criterion chosen.") << Core::Error::abort;
	}
	return res;
}

/*
 * DummyCriterion
 */
DummyCriterion::DummyCriterion() :
		Precursor()
{}

void DummyCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	Core::Error::msg("DummyCriterion: Error signal computation not possible.") << Core::Error::abort;
}

Float DummyCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	return 0.0;
}

/*
 * CrossEntropyCriterion
 */
CrossEntropyCriterion::CrossEntropyCriterion() :
		Precursor()
{}

void CrossEntropyCriterion::sanityCheck(NeuralNetwork& network) {
	if (network.outputLayer().layerType() != Layer::softmax)
		Core::Error::msg("CrossEntropyCriterion: Output layer needs to be a softmax layer.") << Core::Error::abort;
}

void CrossEntropyCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	require_eq(activations.nColumns(), errorSignal.nColumns());
	require_eq(activations.nRows(), errorSignal.nRows());
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	errorSignal.setToZero();
	errorSignal.add(activations);
	errorSignal.add(targets, (Float)-1.0);
}

Float CrossEntropyCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	return activations.crossEntropyObjectiveFunction(targets);
}

/*
 * WeightedCrossEntropyCriterion
 */
const Core::ParameterString WeightedCrossEntropyCriterion::paramWeightVector_("class-weights", "", "training-criterion");

WeightedCrossEntropyCriterion::WeightedCrossEntropyCriterion() :
		Precursor(),
		classWeightsFile_(Core::Configuration::config(paramWeightVector_))
{}

void WeightedCrossEntropyCriterion::initialize(NeuralNetwork& network) {
	require(!classWeightsFile_.empty());
	classWeights_.read(classWeightsFile_);
	require_eq(classWeights_.size(), network.outputDimension());
}

void WeightedCrossEntropyCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	Precursor::initialErrorSignal(activations, targets, errorSignal);
	// fill weightsVector_
	weightsVector_.resize(targets.nColumns());
	weightsVector_.finishComputation(false);
	for (u32 i = 0; i < targets.nColumns(); i++) {
		weightsVector_.at(i) = classWeights_.at(targets.argAbsMax(i));
	}
	weightsVector_.initComputation(true);
	// multiply error signal with weights
	errorSignal.multiplyColumnsByScalars(weightsVector_);
}

Float WeightedCrossEntropyCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	// fill weightsVector_
	weightsVector_.resize(targets.nColumns());
	weightsVector_.finishComputation(false);
	for (u32 i = 0; i < targets.nColumns(); i++) {
		weightsVector_.at(i) = classWeights_.at(targets.argAbsMax(i));
	}
	weightsVector_.initComputation(true);
	return activations.weightedCrossEntropyObjectiveFunction(targets, weightsVector_);
}

/*
 * SquaredErrorCriterion
 */
SquaredErrorCriterion::SquaredErrorCriterion() :
		Precursor(),
		outputLayerType_(Layer::identity)
{}

void SquaredErrorCriterion::sanityCheck(NeuralNetwork& network) {
	switch (network.outputLayer().layerType()) {
	case Layer::sigmoid:
	case Layer::identity:
	case Layer::rectified:
	case Layer::tanh:
		// all these are okay
		outputLayerType_ = network.outputLayer().layerType();
		break;
	default:
		Core::Error::msg("SquaredErrorCriterion: Output layer type not supported in combination with squared error criterion.") << Core::Error::abort;
	}
}

void SquaredErrorCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	require_eq(activations.nColumns(), errorSignal.nColumns());
	require_eq(activations.nRows(), errorSignal.nRows());
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	errorSignal.setToZero();
	errorSignal.add(activations);
	errorSignal.add(targets, (Float)-1.0);
	switch (outputLayerType_) {
	case Layer::sigmoid:
		errorSignal.elementwiseMultiplicationWithSigmoidDerivative(activations);
		break;
	case Layer::rectified:
		errorSignal.elementwiseMultiplicationWithClippedDerivative(activations, 0.0, Types::inf<Float>());
		break;
	case Layer::tanh:
		errorSignal.elementwiseMultiplicationWithTanhDerivative(activations);
		break;
	default:
		; // nothing to do in case of identity layer
	}
}

Float SquaredErrorCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	return activations.squaredErrorObjectiveFunction(targets);
}

/*
 * WeightedSquaredErrorCriterion
 */
const Core::ParameterString WeightedSquaredErrorCriterion::paramWeightVector_("class-weights", "", "training-criterion");

WeightedSquaredErrorCriterion::WeightedSquaredErrorCriterion() :
		Precursor(),
		classWeightsFile_(Core::Configuration::config(paramWeightVector_))
{}

void WeightedSquaredErrorCriterion::initialize(NeuralNetwork& network) {
	require(!classWeightsFile_.empty());
	classWeights_.read(classWeightsFile_);
	require_eq(classWeights_.size(), network.outputDimension());
	classWeights_.initComputation();
}

void WeightedSquaredErrorCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	Precursor::initialErrorSignal(activations, targets, errorSignal);
	require_eq(targets.nRows(), errorSignal.nRows());
	// multiply error signal with weights
	errorSignal.multiplyRowsByScalars(classWeights_);
}

//Float WeightedSquaredErrorCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
//	require_eq(activations.nColumns(), targets.nColumns());
//	require_eq(activations.nRows(), targets.nRows());
//	// fill weights matrix
//	tmpMatrix_.initComputation();
//	tmpMatrix_.resize(activations.nRows(), activations.nColumns());
//	tmpMatrix_.copy(activations);
//	tmpMatrix_.multiplyRowsByScalars(classWeights_);
//	return tmpMatrix_.squaredErrorObjectiveFunction(targets);
//}

/*
 * SmoothedL1Criterion
 */
SmoothedL1Criterion::SmoothedL1Criterion() :
		Precursor(),
		outputLayerType_(Layer::identity)
{}

void SmoothedL1Criterion::sanityCheck(NeuralNetwork& network) {
	switch (network.outputLayer().layerType()) {
	case Layer::sigmoid:
	case Layer::identity:
	case Layer::rectified:
	case Layer::tanh:
		// all these are okay
		outputLayerType_ = network.outputLayer().layerType();
		break;
	default:
		Core::Error::msg("SmoothedL1Criterion: Output layer type not supported in combination with squared error criterion.") << Core::Error::abort;
	}
}

void SmoothedL1Criterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	require_eq(activations.nColumns(), errorSignal.nColumns());
	require_eq(activations.nRows(), errorSignal.nRows());
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	errorSignal.copy(activations);
	errorSignal.add(targets, (Float)-1.0);
	errorSignal.ensureMinimalValue((Float)-1.0);
	errorSignal.ensureMaximalValue((Float)1.0);
	switch (outputLayerType_) {
	case Layer::sigmoid:
		errorSignal.elementwiseMultiplicationWithSigmoidDerivative(activations);
		break;
	case Layer::rectified:
		errorSignal.elementwiseMultiplicationWithClippedDerivative(activations, 0.0, Types::inf<Float>());
		break;
	case Layer::tanh:
		errorSignal.elementwiseMultiplicationWithTanhDerivative(activations);
		break;
	default:
		; // nothing to do in case of identity layer
	}
}

Float SmoothedL1Criterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	return activations.smoothedL1ObjectiveFunction(targets);
}

/*
 * MultiTaskTrainingCriterion
 */
const Core::ParameterInt MultiTaskTrainingCriterion::paramNTasks_("number-of-tasks", 0, "training-criterion");

const Core::ParameterStringList MultiTaskTrainingCriterion::paramOutputLayerNames_("output-layers", "", "training-criterion");

const Core::ParameterFloatList MultiTaskTrainingCriterion::paramTaskWeights_("task-weights", "", "training-criterion");

MultiTaskTrainingCriterion::MultiTaskTrainingCriterion() :
		Precursor(),
		nTasks_(Core::Configuration::config(paramNTasks_)),
		currentTaskIndex_(0),
		requiredTargetDimension_(nTasks_),
		outputLayerNames_(Core::Configuration::config(paramOutputLayerNames_)),
		taskWeights_(Core::Configuration::config(paramTaskWeights_))
{
	if (taskWeights_.size() == 0) {
		taskWeights_.resize(nTasks_, 1.0);
	}
	require_gt(nTasks_, 0);
	require_eq(nTasks_, outputLayerNames_.size());
	require_eq(nTasks_, taskWeights_.size());
}

void MultiTaskTrainingCriterion::initialize(NeuralNetwork& network) {
	taskTargets_.initComputation();
	weightsVector_.initComputation();
	// get output layer types
	for (u32 i = 0; i < outputLayerNames_.size(); i++) {
		outputLayerTypes_.push_back(network.layer(outputLayerNames_.at(i)).layerType());
		requiredTargetDimension_ += network.layer(outputLayerNames_.at(i)).nOutputUnits(0);
		network.layer(outputLayerNames_.at(i)).setAsOutputLayer();
	}
}

void MultiTaskTrainingCriterion::sanityCheck(NeuralNetwork& network) {
	for (u32 i = 0; i < outputLayerTypes_.size(); i++) {
		switch (outputLayerTypes_.at(i)) {
		case Layer::sigmoid:
		case Layer::identity:
		case Layer::rectified:
		case Layer::tanh:
		case Layer::softmax:
			break;
		default:
			Core::Error::msg("MultiTaskTrainingCriterion: Type of output layer ") << outputLayerNames_.at(i) << " not supported in combination with multi-task-training-criterion." << Core::Error::abort;
		}
	}
}

void MultiTaskTrainingCriterion::initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) {
	require_eq(activations.nColumns(), errorSignal.nColumns());
	require_eq(activations.nRows(), errorSignal.nRows());
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());

	errorSignal.setToZero();
	errorSignal.add(activations);
	errorSignal.add(targets, (Float)-1.0);
	switch (outputLayerTypes_.at(currentTaskIndex_)) {
	case Layer::sigmoid:
		errorSignal.elementwiseMultiplicationWithSigmoidDerivative(activations);
		break;
	case Layer::rectified:
		errorSignal.elementwiseMultiplicationWithClippedDerivative(activations, 0.0, Types::inf<Float>());
		break;
	case Layer::tanh:
		errorSignal.elementwiseMultiplicationWithTanhDerivative(activations);
		break;
	default:
		; // nothing to do in case of identity and softmax layer
	}
	errorSignal.multiplyColumnsByScalars(weightsVector_);
}

Float MultiTaskTrainingCriterion::objectiveFunction(const Matrix& activations, const Matrix& targets) {
	require_eq(activations.nColumns(), targets.nColumns());
	require_eq(activations.nRows(), targets.nRows());
	switch (outputLayerTypes_.at(currentTaskIndex_)) {
	case Layer::sigmoid:
	case Layer::rectified:
	case Layer::tanh:
	case Layer::identity:
		return activations.weightedSquaredErrorObjectiveFunction(targets, weightsVector_);
		break;
	case Layer::softmax:
		return activations.weightedCrossEntropyObjectiveFunction(targets, weightsVector_);
		break;
	default:
		return 0; // can not happen
	}
}

void MultiTaskTrainingCriterion::computeInitialErrorSignal(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	require_eq(targets.nRows(), requiredTargetDimension_);
	u32 targetRowIndex = nTasks_;
	// compute initial error signal for the respective task
	for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
		// copy relevant part of target matrix to taskTargets
		taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.nColumns());
		taskTargets_.copyBlockFromMatrix(targets, targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
		// compute actual error signal
		targets.getRow(currentTaskIndex_, weightsVector_);
		weightsVector_.scale(taskWeights_.at(currentTaskIndex_));
		initialErrorSignal(network.layer(outputLayerNames_.at(currentTaskIndex_)).latestActivations(0), taskTargets_,
				network.layer(outputLayerNames_.at(currentTaskIndex_)).latestErrorSignal(0));
		// store the offset (at which index the targets for the current task start)
		targetRowIndex += taskTargets_.nRows();
	}
}

void MultiTaskTrainingCriterion::computeInitialErrorSignals(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		require_eq(targets.at(t).nRows(), requiredTargetDimension_);
		u32 targetRowIndex = nTasks_;
		// compute initial error signal for the respective task
		for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
			// copy relevant part of target matrix to taskTargets
			taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.at(t).nColumns());
			taskTargets_.copyBlockFromMatrix(targets.at(t), targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
			// compute actual error signal
			targets.at(t).getRow(currentTaskIndex_, weightsVector_);
			weightsVector_.scale(taskWeights_.at(currentTaskIndex_));
			initialErrorSignal(network.layer(outputLayerNames_.at(currentTaskIndex_)).activations(t, 0), taskTargets_,
					network.layer(outputLayerNames_.at(currentTaskIndex_)).errorSignal(t, 0));
			// store the offset (at which index the targets for the current task start)
			targetRowIndex += taskTargets_.nRows();
		}
	}
}

Float MultiTaskTrainingCriterion::computeObjectiveFunction(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	require_eq(targets.nRows(), requiredTargetDimension_);
	u32 targetRowIndex = nTasks_;
	float objFctn = 0;
	// compute objective function for the respective task
	for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
		// copy relevant part of target matrix to taskTargets
		taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.nColumns());
		taskTargets_.copyBlockFromMatrix(targets, targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
		// compute actual objective function
		targets.getRow(currentTaskIndex_, weightsVector_);
		weightsVector_.scale(taskWeights_.at(currentTaskIndex_));
		objFctn += objectiveFunction(network.layer(outputLayerNames_.at(currentTaskIndex_)).latestActivations(0), taskTargets_);
		// store the offset (at which index the targets for the current task start)
		targetRowIndex += taskTargets_.nRows();
	}
	return objFctn;
}

Float MultiTaskTrainingCriterion::computeObjectiveFunction(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	float objFctn = 0;
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		require_eq(targets.at(t).nRows(), requiredTargetDimension_);
		u32 targetRowIndex = nTasks_;
		// compute objective function for the respective task
		for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
			// copy relevant part of target matrix to taskTargets
			taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.at(t).nColumns());
			taskTargets_.copyBlockFromMatrix(targets.at(t), targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
			// compute actual objective function
			targets.at(t).getRow(currentTaskIndex_, weightsVector_);
			weightsVector_.scale(taskWeights_.at(currentTaskIndex_));
			objFctn += objectiveFunction(network.layer(outputLayerNames_.at(currentTaskIndex_)).activations(t, 0), taskTargets_);
			// store the offset (at which index the targets for the current task start)
			targetRowIndex += taskTargets_.nRows();
		}
	}
	return objFctn;
}

u32 MultiTaskTrainingCriterion::nClassificationErrors(NeuralNetwork& network, const Matrix& targets) {
	sanityCheck(network);
	require_eq(targets.nRows(), requiredTargetDimension_);
	u32 targetRowIndex = nTasks_;
	u32 classificationErrors = 0;
	// compute number of classification errors for the respective task
	for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
		if (network.layer(outputLayerNames_.at(currentTaskIndex_)).layerType() == Layer::softmax) {
			// copy relevant part of target matrix to taskTargets
			taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.nColumns());
			taskTargets_.copyBlockFromMatrix(targets, targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
			classificationErrors += network.layer(outputLayerNames_.at(currentTaskIndex_)).latestActivations(0).nClassificationErrors(taskTargets_);
			// store the offset (at which index the targets for the current task start)
			targetRowIndex += taskTargets_.nRows();
		}
	}
	return classificationErrors;
}

u32 MultiTaskTrainingCriterion::nClassificationErrors(NeuralNetwork& network, const MatrixContainer& targets) {
	sanityCheck(network);
	u32 classificationErrors = 0;
	for (u32 t = 0; t < targets.nTimeframes(); t++) {
		require_eq(targets.at(t).nRows(), requiredTargetDimension_);
		u32 targetRowIndex = nTasks_;
		// compute number of classification errors for the respective task
		for (currentTaskIndex_ = 0; currentTaskIndex_ < nTasks_; currentTaskIndex_++) {
			if (network.layer(outputLayerNames_.at(currentTaskIndex_)).layerType() == Layer::softmax) {
				// copy relevant part of target matrix to taskTargets
				taskTargets_.resize(network.layer(outputLayerNames_.at(currentTaskIndex_)).nOutputUnits(0), targets.at(t).nColumns());
				taskTargets_.copyBlockFromMatrix(targets.at(t), targetRowIndex, 0, 0, 0, taskTargets_.nRows(), taskTargets_.nColumns());
				classificationErrors += network.layer(outputLayerNames_.at(currentTaskIndex_)).activations(t, 0).nClassificationErrors(taskTargets_);
				// store the offset (at which index the targets for the current task start)
				targetRowIndex += taskTargets_.nRows();
			}
		}
	}
	return classificationErrors;
}
