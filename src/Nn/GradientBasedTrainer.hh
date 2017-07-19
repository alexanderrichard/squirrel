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
 * GradientBasedTrainer.hh
 *
 *  Created on: May 21, 2014
 *      Author: richard
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
	u32 epochAtLastUpdate_;
	u32 firstTrainableLayerIndex_;
	u32 lastRecurrentLayerIndex_;

	// specify which statistics to store
	virtual u32 requiredStatistics();

	// access methods of training criterion
	virtual void computeInitialErrorSignal(Matrix& targets) { criterion_->computeInitialErrorSignal(network(), targets); }
	virtual void computeInitialErrorSignals(MatrixContainer& targets) { criterion_->computeInitialErrorSignals(network(), targets); }
	virtual Float computeObjectiveFunction(Matrix& targets) { return criterion_->computeObjectiveFunction(network(), targets); }
	virtual Float computeObjectiveFunction(MatrixContainer& targets) { return criterion_->computeObjectiveFunction(network(), targets); }
	virtual u32 nClassificationErrors(Matrix& targets) { return criterion_->nClassificationErrors(network(), targets); }
	virtual u32 nClassificationErrors(MatrixContainer& targets) { return criterion_->nClassificationErrors(network(), targets); }

	/*
	 * @param t the timeframe for which to do compute the error signals
	 * @param layerIndexFrom, layerIndexTo compute error signals only for layers [layerIndexFrom, layerIndexTo]
	 * @param greedyBackprop if current timeframe is not sequence end only backprop from last recurrent layer on. Gradient is only exact if target is not a sequence.
	 */
	virtual void backpropTimeframe(u32 t, u32 layerIndexFrom = 0, u32 layerIndexTo = Types::max<u32>(), bool greedyBackprop = true);
	/*
	 * @param historyLength the number of timeframes to be stored for gradient computation (less than all -> approximate gradient)
	 *        historyLength = 1 is the default value for feed forward training (only the most recent frame is memorized)
	 * @param greedyBackprop if current timeframe is not sequence end only backprop from last recurrent layer on. Gradient is only exact if target is not a sequence.
	 */
	virtual void errorBackpropagation(u32 historyLength = 1, bool greedyBackprop = true);
	virtual void updateGradient(u32 historyLength = 1);
	virtual void estimateModelParameters();
public:
	GradientBasedTrainer();
	virtual ~GradientBasedTrainer();
	virtual void initialize();

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
	virtual void initialize();
	virtual void processBatch(Matrix& source, Matrix& targets);
	// process each element of the sequence one by one but recurrency in forwarding is possible
	// (training with truncated gradient, only current frame regarded for gradient)
	virtual void processSequenceBatch(MatrixContainer& source, MatrixContainer& target);
};

/*
 * neural network trainer for recurrent neural networks
 */
class RnnTrainer : public GradientBasedTrainer
{
private:
	typedef GradientBasedTrainer Precursor;
	static const Core::ParameterInt paramMaxTimeHistory_;
	static const Core::ParameterBool paramGreedyForwarding_;
protected:
	u32 maxTimeHistory_;
	bool greedyForwarding_;
	virtual u32 requiredStatistics();
public:
	RnnTrainer();
	virtual ~RnnTrainer() {}
	virtual void initialize();
	virtual void processSequenceBatch(MatrixContainer& source, Matrix& targets);
	virtual void processSequenceBatch(MatrixContainer& source, MatrixContainer& targets);
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
	virtual void initialize();
	virtual void processSequenceBatch(MatrixContainer& source, Matrix& targets);
};

class SpecialRnnTrainer : public RnnTrainer
{
private:
	typedef RnnTrainer Precursor;
	static const Core::ParameterInt paramEffectiveBatchSize_;
	static const Core::ParameterInt paramUpdateAfter_;
private:
	u32 effectiveBatchSize_;
	u32 currentFrameIdx_;
	u32 currentSequenceIdx_;
	MatrixContainer batchedSource_;
	Matrix batchedTargets_;
	void setEffectiveContainerSize(u32 sourceDim, u32 targetDim, u32 batchSize);
public:
	SpecialRnnTrainer();
	virtual ~SpecialRnnTrainer() {}
	virtual void processSequenceBatch(MatrixContainer& source, Matrix& targets);
	void processSequenceBatch(MatrixContainer& source, MatrixContainer& targets);
};

} // namespace

#endif /* NN_GRADIENTBASEDTRAINER_HH_ */
