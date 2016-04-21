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

#include "GradientBasedTrainer.hh"

using namespace Nn;
using namespace std;

/*
 * GradientBasedTrainer
 */
const Core::ParameterEnum GradientBasedTrainer::paramModelUpdateStrategy_("model-update-strategy",
		"after-batch, after-epoch", "after-batch", "trainer");

GradientBasedTrainer::GradientBasedTrainer() :
		Precursor(),
		statistics_(0),
		regularizer_(0),
		criterion_(0),
		modelUpdateStrategy_((ModelUpdateStrategy) Core::Configuration::config(paramModelUpdateStrategy_)),
		firstTrainableLayerIndex_(0),
		lastRecurrentLayerIndex_(0)
{}

GradientBasedTrainer::~GradientBasedTrainer() {
	if (regularizer_)
		delete regularizer_;
}

void GradientBasedTrainer::initialize(u32 epochLength) {
	Precursor::initialize(epochLength);
	/* find index of first layer with trainable input connection or trainable bias */
	firstTrainableLayerIndex_ = network().nLayer();
	for (s32 l = network().nLayer() - 1; l >= 0; l--) {
		for (u32 port = 0; port < network().layer(l).nInputPorts(); port++) {
			for (u32 c = 0; c < network().layer(l).nIncomingConnections(port); c++) {
				if (network().layer(l).isTrainable(c, port)) {
					firstTrainableLayerIndex_ = (u32)l;
					break;
				}
			}
		}
		if (network().layer(l).useBias() && network().layer(l).isBiasTrainable())
			firstTrainableLayerIndex_ = (u32)l;
	}
	// error signals need to be computed for all layers from the first trainable layer on
	network().setTrainingMode(true, firstTrainableLayerIndex_);
	statistics_ = new Statistics((Statistics::StatisticTypes)estimator().requiredStatistics());
	statistics().initialize(network());
	regularizer_ = Regularizer::createRegularizer();
	criterion_ = TrainingCriterion::createCriterion();
	criterion_->initialize(network());
	lastRecurrentLayerIndex_ = std::min(network().lastRecurrentLayerIndex(), network().nLayer() - 1);
}

void GradientBasedTrainer::errorBackpropagation(u32 historyLength) {
	/* backpropagation through time */
	u32 T = network().outputLayer().nTimeframes() - 1;
	for (s32 t = T; t > (s32)T - (s32)historyLength; t--) {
		/* hide all error signals of sequences that begin in time frame t+1 */
		if (t < (s32)T) {
			for (u32 l = 0; l < network().nLayer(); l++) {
				if (network().layer(l).isInTrainingMode()) {
					network().layer(l).setErrorSignalVisibility(t+1, network().layer(l).activations(t,0).nColumns());
					network().layer(l).setErrorSignalVisibility(t, network().layer(l).activations(t,0).nColumns());
				}
			}
		}
		/* backpropagation through the layers */
		// if t == T: exclude layer L because e(L,T) is the initial error signal
		// if t < T: backpropagation only from last recurrent layer on
		s32 L;
		if (t == (s32)T)
			L = network().nLayer() - 2;
		else
			L = lastRecurrentLayerIndex_;
		for (s32 l = L; l >= (s32)firstTrainableLayerIndex_; l--) {
			require_lt((u32)t, network().layer(l).nTimeframes());
			network().layer(l).backpropagate(t);
		}
		/* undo the hiding */
		if (t < (s32)T) {
			for (u32 l = 0; l < network().nLayer(); l++) {
				if (network().layer(l).isInTrainingMode()) {
					network().layer(l).setErrorSignalVisibility(t+1, network().layer(l).activations(t+1,0).nColumns());
				}
			}
		}
	}
}

void GradientBasedTrainer::updateGradient(u32 historyLength, u32 maxLayerIndex) {
	u32 T = network().outputLayer().nTimeframes() - 1;

	// update the weight gradients
	for (u32 c = 0; c < network().nConnections(); c++) {
		if (network().connection(c).hasWeights() && network().connection(c).isTrainable()) {
			u32 sourcePort = network().connection(c).sourcePort();
			u32 destPort = network().connection(c).destinationPort();
			// sum over the complete stored history of time frames
			for (s32 t = T; t > (s32)T - (s32)historyLength; t--) {
				// if error signal exists (might not be the case if t < T and connection is beyond last recurrent layer)
				if (network().connection(c).to().errorSignal(t, destPort).nRows() > 0) {
					statistics().weightsGradient(network().connection(c).name()).addMatrixProduct(
							network().connection(c).from().activations(t, sourcePort),
							network().connection(c).to().errorSignal(t, destPort),
							1.0, 1.0, false, true);
				}
			}
		}
	}
	// update the bias gradients
	for (u32 l = 0; l < network().nLayer(); l++) {
		if (network().layer(l).useBias() && network().layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network().layer(l).nInputPorts(); port++) {
				// sum over the complete stored history of time frames
				for (s32 t = T; t > (s32)T - (s32)historyLength; t--) {
					// if error signal exists (might not be the case if t < T and layer is beyond last recurrent layer)
					if (network().layer(l).errorSignal(t, port).nRows() > 0) {
						statistics().biasGradient(network().layer(l).name(), port).addSummedColumns(
								network().layer(l).errorSignal(t, port));
					}
				}
			}
		}
	}
}

void GradientBasedTrainer::estimateModelParameters() {
	statistics().normalize();
	// add the regularizer gradient and objective function
	regularizer_->addToGradient(network(), statistics());
	regularizer_->addToObjectiveFunction(network(), statistics());

	/* estimate parameters */
	estimator_->estimate(network(), statistics());

	/* log objective function */
	Core::Log::os("objective function value: ") << statistics().objectiveFunction();
	Core::Log::os("classification error rate: ") << (Float)statistics().nClassificationErrors() / statistics().nObservations();

	/* reset the statistics */
	statistics().reset();
}

/*
 * FeedForwardTrainer
 */
FeedForwardTrainer::FeedForwardTrainer() :
		Precursor()
{}

void FeedForwardTrainer::initialize(u32 epochLength) {
	Precursor::initialize(epochLength);
	require(!network_.isRecurrent());
	// set the history length (number of time frames that are memorized) to 1
	network().setMaximalMemory(1);
}

void FeedForwardTrainer::processBatch(Matrix& batch, LabelVector& labels) {
	require_eq(batch.nColumns(), labels.nRows());
	/* initial steps for the batch */
	// ensure everything is in computing state
	labels.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* forward the network */
	network().forward(batch);

	/* compute initial error signal */
	computeInitialErrorSignal(labels);

	/* error backpropagation */
	errorBackpropagation();

	/* update statistics */
	Float objFctn = computeObjectiveFunction(labels);
	statistics().addToObjectiveFunction(objFctn);
	statistics().increaseNumberOfClassificationErrors(network().outputLayer().latestActivations(0).nClassificationErrors(labels));
	statistics().increaseNumberOfObservations(batch.nColumns());
	updateGradient();

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}

/*
 * RnnTrainer
 */
const Core::ParameterInt RnnTrainer::paramMaxTimeHistory_("max-time-history", Types::max<u32>(), "trainer");

RnnTrainer::RnnTrainer() :
		Precursor(),
		maxTimeHistory_(Core::Configuration::config(paramMaxTimeHistory_))
{}

void RnnTrainer::initialize(u32 epochLength) {
	require(maxTimeHistory_ > 0);
	Precursor::initialize(epochLength);
}

void RnnTrainer::processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels) {
	require_eq(batchedSequence.getLast().nColumns(), labels.nRows());

	/* initial steps for the sequence batch */
	// set the history length (number of time frames that are memorized)
	u32 historyLength = (maxTimeHistory_ < batchedSequence.nTimeframes() ? maxTimeHistory_ : batchedSequence.nTimeframes());
	network().setMaximalMemory(historyLength);
	// ensure everything is in computing state
	labels.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* forward the network */
	network().forwardSequence(batchedSequence);

	/* compute initial error signal */
	computeInitialErrorSignal(labels);

	/* error backpropagation */
	errorBackpropagation(historyLength);

	/* update statistics */
	Float objFctn = computeObjectiveFunction(labels);
	statistics().addToObjectiveFunction(objFctn);
	statistics().increaseNumberOfClassificationErrors(network().outputLayer().latestActivations(0).nClassificationErrors(labels));
	statistics().increaseNumberOfObservations(batchedSequence.getLast().nColumns());
	updateGradient(historyLength);

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}
