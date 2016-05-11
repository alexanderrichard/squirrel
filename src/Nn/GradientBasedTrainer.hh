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

#ifndef NN_GRADIENTBASEDTRAINER_HH_
#define NN_GRADIENTBASEDTRAINER_HH_

#include <Core/CommonHeaders.hh>
#include "Trainer.hh"
#include "Statistics.hh"
#include "Regularizer.hh"
#include "TrainingCriteria.hh"

namespace Nn {

/*
 * base class for gradient based neural network trainers
 * includes error backpropagation and gradient update
 */
class GradientBasedTrainer : public Trainer
{
private:
	typedef Trainer Precursor;
	static const Core::ParameterEnum paramModelUpdateStrategy_;
protected:
	enum ModelUpdateStrategy { afterBatch, afterEpoch };
	Statistics* statistics_;
	Regularizer* regularizer_;
	TrainingCriterion* criterion_;
	ModelUpdateStrategy modelUpdateStrategy_;
	u32 firstTrainableLayerIndex_;
	u32 lastRecurrentLayerIndex_;

	// access methods of training criterion
	virtual void computeInitialErrorSignal(LabelVector& labels) { criterion_->computeInitialErrorSignal(network(), labels); }
	virtual Float computeObjectiveFunction(LabelVector& labels) { return criterion_->computeObjectiveFunction(network(), labels); }
	// historyLength = 1 is the default value for feed forward training (only the most recent frame is memorized)
	virtual void errorBackpropagation(u32 historyLength = 1);
	virtual void updateGradient(u32 historyLength = 1, u32 maxLayerIndex = Types::max<u32>());
	virtual void estimateModelParameters();
public:
	GradientBasedTrainer();
	virtual ~GradientBasedTrainer();
	virtual void initialize(u32 epochLength = 0);
	virtual bool isSupervised() const { return true; }
	virtual bool isUnsupervised() const { return false; }

	virtual Statistics& statistics() { return *statistics_; }
};

/*
 * neural network trainer for feed forward neural networks
 */
class FeedForwardTrainer : public GradientBasedTrainer
{
private:
	typedef GradientBasedTrainer Precursor;
public:
	FeedForwardTrainer();
	virtual ~FeedForwardTrainer() {}
	virtual void initialize(u32 epochLength = 0);
	/*
	 * compute the gradient for the features in the current batch
	 * @param batch contains the features
	 * @param labels contains the corresponding labels
	 */
	virtual void processBatch(Matrix& batch, LabelVector& labels);
};

/*
 * neural network trainer for recurrent neural networks
 */
class RnnTrainer : public GradientBasedTrainer
{
private:
	typedef GradientBasedTrainer Precursor;
	static const Core::ParameterInt paramMaxTimeHistory_;
private:
	u32 maxTimeHistory_;
public:
	RnnTrainer();
	virtual ~RnnTrainer() {}
	virtual void initialize(u32 epochLength = 0);
	/*
	 * compute the gradient for the feature sequences in the current batch
	 * @param batch contains the feature sequences (one matrix corresponds to one sequence)
	 * @param labels contains the corresponding labels
	 */
	virtual void processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels);
};

/*
 * neural network trainer for bag-of-words neural networks (networks with just one recurrent identity/sequence-length-normalization layer)
 */
class BagOfWordsNetworkTrainer : public GradientBasedTrainer
{
private:
	typedef GradientBasedTrainer Precursor;
private:
	std::vector<Matrix> recurrentErrorSignal_;	// store recurrent error signal for each port of the recurrent layer
	u32 recurrentLayerIndex_;
	virtual void framewiseErrorBackpropagation();
	virtual void _updateGradient();
public:
	BagOfWordsNetworkTrainer();
	virtual ~BagOfWordsNetworkTrainer() {}
	virtual void initialize(u32 epochLength = 0);
	/*
	 * compute the gradient for the feature sequences in the current batch
	 * @param batch contains the feature sequences (one matrix corresponds to one sequence)
	 * @param labels contains the corresponding labels
	 */
	virtual void processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels);
};

} // namespace

#endif /* NN_GRADIENTBASEDTRAINER_HH_ */
