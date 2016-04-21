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

#include "TrainingCriteria.hh"
#include "ActivationLayer.hh"

using namespace Nn;

/*
 * TrainingCriterion
 */
const Core::ParameterEnum TrainingCriterion::paramTrainingCriterion_("training-criterion",
		"none, cross-entropy, weighted-cross-entropy, squared-error", "none", "");

TrainingCriterion* TrainingCriterion::createCriterion() {
	TrainingCriterion* res = 0;
	switch ((CriterionType) Core::Configuration::config(paramTrainingCriterion_)) {
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
		default:
			std::cerr << "TrainingCriterion.cc: no criterion chosen. Abort." << std::endl;
	}
	return res;
}

/*
 * CrossEntropyCriterion
 */
void CrossEntropyCriterion::computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels) {
	if (network.outputLayer().layerType() == Layer::softmax) {
		network.outputLayer().latestErrorSignal(0).setToZero();
		network.outputLayer().latestErrorSignal(0).add(network.outputLayer().latestActivations(0));
		network.outputLayer().latestErrorSignal(0).addKroneckerDelta(labels, -1.0);
		SoftmaxLayer* tmp = dynamic_cast<SoftmaxLayer*>(&(network.outputLayer()));
		if (tmp->enhancementFactor() != 1.0) {
			network.outputLayer().latestErrorSignal(0).scale(tmp->enhancementFactor());
		}
	}
	else {
		std::cerr << "CrossEntropyCriterion: Output layer needs to be a softmax layer. Abort." << std::endl;
		exit(1);
	}
}

Float CrossEntropyCriterion::computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels) {
	Float objFctn = 0;
	if (network.outputLayer().layerType() == Layer::softmax) {
		objFctn =  network.outputLayer().latestActivations(0).crossEntropyObjectiveFunction(labels);
	}
	else {
		std::cerr << "CrossEntropyCriterion: Output layer needs to be a softmax layer. Abort." << std::endl;
		exit(1);
	}
	return objFctn;
}

/*
 * WeightedCrossEntropyCriterion
 */
const Core::ParameterString WeightedCrossEntropyCriterion::paramWeightVector_("class-weights", "", "");

WeightedCrossEntropyCriterion::WeightedCrossEntropyCriterion() :
		TrainingCriterion(),
		classWeightsFile_(Core::Configuration::config(paramWeightVector_))
{}

void WeightedCrossEntropyCriterion::initialize(NeuralNetwork& network) {
	require(!classWeightsFile_.empty());
	classWeights_.read(classWeightsFile_);
	require_eq(classWeights_.size(), network.outputDimension());
}

void WeightedCrossEntropyCriterion::computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels) {
	if (network.outputLayer().layerType() == Layer::softmax) {
		network.outputLayer().latestErrorSignal(0).setToZero();
		network.outputLayer().latestErrorSignal(0).add(network.outputLayer().latestActivations(0));
		network.outputLayer().latestErrorSignal(0).addKroneckerDelta(labels, -1.0);
		// fill weightsVector_
		labels.finishComputation(false);
		weightsVector_.resize(labels.size());
		weightsVector_.finishComputation(false);
		for (u32 i = 0; i < labels.size(); i++) {
			weightsVector_.at(i) = classWeights_.at(labels.at(i));
		}
		labels.initComputation(false);
		weightsVector_.initComputation(true);
		network.outputLayer().latestErrorSignal(0).multiplyColumnsByScalars(weightsVector_);
		SoftmaxLayer* tmp = dynamic_cast<SoftmaxLayer*>(&(network.outputLayer()));
		if (tmp->enhancementFactor() != 1.0) {
			network.outputLayer().latestErrorSignal(0).scale(tmp->enhancementFactor());
		}
	}
	else {
		std::cerr << "WeightedCrossEntropyCriterion: Output layer needs to be a softmax layer. Abort." << std::endl;
		exit(1);
	}
}

Float WeightedCrossEntropyCriterion::computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels) {
	Float objFctn = 0;
	if (network.outputLayer().layerType() == Layer::softmax) {
		// fill weightsVector_
		labels.finishComputation(false);
		weightsVector_.resize(labels.size());
		weightsVector_.finishComputation(false);
		for (u32 i = 0; i < labels.size(); i++) {
			weightsVector_.at(i) = classWeights_.at(labels.at(i));
		}
		labels.initComputation(false);
		weightsVector_.initComputation(true);
		objFctn =  network.outputLayer().latestActivations(0).weightedCrossEntropyObjectiveFunction(labels, weightsVector_);
	}
	else {
		std::cerr << "WeightedCrossEntropyCriterion: Output layer needs to be a softmax layer. Abort." << std::endl;
		exit(1);
	}
	return objFctn;
}

/*
 * SquaredErrorCriterion
 */
void SquaredErrorCriterion::computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels) {
	if ((network.outputLayer().layerType() == Layer::sigmoid) || (network.outputLayer().layerType() == Layer::identity)) {
		network.outputLayer().latestErrorSignal(0).setToZero();
		network.outputLayer().latestErrorSignal(0).add(network.outputLayer().latestActivations(0));
		network.outputLayer().latestErrorSignal(0).addKroneckerDelta(labels, -1.0);
	}
	else {
		std::cerr << "SquaredErrorCriterion: Output layer needs to be an identity or sigmoid layer. Abort." << std::endl;
		exit(1);
	}
}

Float SquaredErrorCriterion::computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels) {
	Float objFctn = 0;
	if ((network.outputLayer().layerType() == Layer::sigmoid) || (network.outputLayer().layerType() == Layer::identity)) {
		objFctn =  network.outputLayer().latestActivations(0).squaredErrorObjectiveFunction(labels);
	}
	else {
		std::cerr << "SquaredErrorCriterion: Output layer needs to be an identity or sigmoid layer. Abort." << std::endl;
		exit(1);
	}
	return objFctn;
}
