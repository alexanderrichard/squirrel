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
 * TrainingCriteria.hh
 *
 *  Created on: Jun 5, 2014
 *      Author: richard
 */

#ifndef NN_TRAININGCRITERIA_HH_
#define NN_TRAININGCRITERIA_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "NeuralNetwork.hh"

namespace Nn {

/*
 * base class for training criteria
 */
class TrainingCriterion
{
private:
	static const Core::ParameterEnum paramTrainingCriterion_;
public:
	enum CriterionType { none, dummy, crossEntropy, weightedCrossEntropy, squaredError, weightedSquaredError, smoothedL1, multiTaskTraining };
protected:
	virtual void sanityCheck(NeuralNetwork& network) {}
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal) = 0;
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets) = 0;
public:
	TrainingCriterion() {}
	virtual ~TrainingCriterion() {}
	virtual void initialize(NeuralNetwork& network) {}
	// initial error signal for single targets
	virtual void computeInitialErrorSignal(NeuralNetwork& network, const Matrix& targets);
	// initial error signal for target sequences
	virtual void computeInitialErrorSignals(NeuralNetwork& network, const MatrixContainer& targets);
	// objective function for single targets
	virtual Float computeObjectiveFunction(NeuralNetwork& network, const Matrix& targets);
	// objective function for sequence targets
	virtual Float computeObjectiveFunction(NeuralNetwork& network, const MatrixContainer& targets);
	// number of classification errors for single targets
	virtual u32 nClassificationErrors(NeuralNetwork& network, const Matrix& targets);
	// number of classification errors for sequence targets
	virtual u32 nClassificationErrors(NeuralNetwork& network, const MatrixContainer& targets);

	static TrainingCriterion* createCriterion();
};

/*
 * dummy criterion: just compute classification error, no loss
 */
class DummyCriterion : public TrainingCriterion
{
private:	virtual void sanityCheck(NeuralNetwork& network) {}
	typedef TrainingCriterion Precursor;
protected:
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	DummyCriterion();
	virtual ~DummyCriterion() {}
};

/*
 * cross-entropy criterion
 */
class CrossEntropyCriterion : public TrainingCriterion
{
private:
	typedef TrainingCriterion Precursor;
protected:
	virtual void sanityCheck(NeuralNetwork& network);
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	CrossEntropyCriterion();
	virtual ~CrossEntropyCriterion() {}
};

/*
 * weighted cross-entropy criterion
 */
class WeightedCrossEntropyCriterion : public CrossEntropyCriterion
{
private:
	static const Core::ParameterString paramWeightVector_;
	typedef CrossEntropyCriterion Precursor;
protected:
	std::string classWeightsFile_;
	Math::Vector<Float> classWeights_;
	Vector weightsVector_;
protected:
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	WeightedCrossEntropyCriterion();
	virtual ~WeightedCrossEntropyCriterion() {}
	virtual void initialize(NeuralNetwork& network);
};

/*
 * squared-error criterion
 */
class SquaredErrorCriterion : public TrainingCriterion
{
private:
	typedef TrainingCriterion Precursor;
protected:
	Layer::LayerType outputLayerType_;
	virtual void sanityCheck(NeuralNetwork& network);
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	SquaredErrorCriterion();
	virtual ~SquaredErrorCriterion() {}
};

/*
 * weighted squared-error criterion
 */
class WeightedSquaredErrorCriterion : public SquaredErrorCriterion
{
private:
	static const Core::ParameterString paramWeightVector_;
	typedef SquaredErrorCriterion Precursor;
protected:
	std::string classWeightsFile_;
	Vector classWeights_;
	Matrix tmpMatrix_;
protected:
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	//virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	WeightedSquaredErrorCriterion();
	virtual ~WeightedSquaredErrorCriterion() {}
	virtual void initialize(NeuralNetwork& network);
};

/*
 * smoothed-l1 criterion
 * squared error in [-1,1], else l1-loss (gradient always between -1 and 1)
 */
class SmoothedL1Criterion : public TrainingCriterion
{
private:
	typedef TrainingCriterion Precursor;
protected:
	Layer::LayerType outputLayerType_;
	virtual void sanityCheck(NeuralNetwork& network);
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	SmoothedL1Criterion();
	virtual ~SmoothedL1Criterion() {}
};

/*
 * multi-task training criterion
 *
 * for softmax output layer, cross entropy is used
 * for sigmoid, identity, tanh, and rectified output layers, squared error is used
 *
 * targets are expected to be in the following form:
 * number-of-tasks entries specifying the weight of the current observation for task n (usually 0 or 1)
 * followed by layer(outputLayerName(n)).nOutputUnits() entries specifying the ground truth for task n
 */
class MultiTaskTrainingCriterion : public TrainingCriterion
{
private:
	typedef TrainingCriterion Precursor;
	static const Core::ParameterInt paramNTasks_;
	static const Core::ParameterStringList paramOutputLayerNames_;
	static const Core::ParameterFloatList paramTaskWeights_;
protected:
	u32 nTasks_;
	u32 currentTaskIndex_;
	u32 requiredTargetDimension_;
	std::vector<std::string> outputLayerNames_;
	std::vector<Layer::LayerType> outputLayerTypes_;
	std::vector<Float> taskWeights_;
	Matrix taskTargets_;
	Vector weightsVector_;
	virtual void sanityCheck(NeuralNetwork& network);
	virtual void initialErrorSignal(const Matrix& activations, const Matrix& targets, Matrix& errorSignal);
	virtual Float objectiveFunction(const Matrix& activations, const Matrix& targets);
public:
	MultiTaskTrainingCriterion();
	virtual ~MultiTaskTrainingCriterion() {}
	virtual void initialize(NeuralNetwork& network);
	// initial error signal for single targets
	virtual void computeInitialErrorSignal(NeuralNetwork& network, const Matrix& targets);
	// initial error signal for target sequences
	virtual void computeInitialErrorSignals(NeuralNetwork& network, const MatrixContainer& targets);
	// objective function for single targets
	virtual Float computeObjectiveFunction(NeuralNetwork& network, const Matrix& targets);
	// objective function for sequence targets
	virtual Float computeObjectiveFunction(NeuralNetwork& network, const MatrixContainer& targets);
	// number of classification errors for single targets
	virtual u32 nClassificationErrors(NeuralNetwork& network, const Matrix& targets);
	// number of classification errors for sequence targets
	virtual u32 nClassificationErrors(NeuralNetwork& network, const MatrixContainer& targets);
};

} // namespace

#endif /* NN_TRAININGCRITERIA_HH_ */
