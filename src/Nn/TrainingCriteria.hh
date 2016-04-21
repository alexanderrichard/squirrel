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
	enum CriterionType { none, crossEntropy, weightedCrossEntropy, squaredError };
public:
	TrainingCriterion() {}
	virtual ~TrainingCriterion() {}
	virtual void initialize(NeuralNetwork& network) {}
	/*
	 * compute the initial error signal at the output layer for the chosen training criterion
	 * @param network the neural network object the trainer operates on
	 * @param labels the (ground truth) labels used for training
	 */
	virtual void computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels) = 0;
	/*
	 * compute the objective function value for the chosen training criterion
	 * @param network the neural network object the trainer operates on
	 * @param labels the (ground truth) labels used for training
	 */
	virtual Float computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels) = 0;

	static TrainingCriterion* createCriterion();
};

/*
 * cross-entropy criterion
 */
class CrossEntropyCriterion : public TrainingCriterion
{
public:
	CrossEntropyCriterion() {}
	virtual ~CrossEntropyCriterion() {}
	virtual void computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels);
	virtual Float computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels);
};

/*
 * weighted cross-entropy criterion
 */
class WeightedCrossEntropyCriterion : public TrainingCriterion
{
private:
	static const Core::ParameterString paramWeightVector_;
	std::string classWeightsFile_;
	Math::Vector<Float> classWeights_;
	Vector weightsVector_;
public:
	WeightedCrossEntropyCriterion();
	virtual ~WeightedCrossEntropyCriterion() {}
	virtual void initialize(NeuralNetwork& network);
	virtual void computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels);
	virtual Float computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels);
};

/*
 * squared-error criterion
 */
class SquaredErrorCriterion : public TrainingCriterion
{
public:
	SquaredErrorCriterion() {}
	virtual ~SquaredErrorCriterion() {}
	virtual void computeInitialErrorSignal(NeuralNetwork& network, LabelVector& labels);
	virtual Float computeObjectiveFunction(NeuralNetwork& network, LabelVector& labels);
};

} // namespace

#endif /* NN_TRAININGCRITERIA_HH_ */
