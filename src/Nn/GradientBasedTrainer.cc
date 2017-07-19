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
 * GradientBasedTrainer.cc
 *
 *  Created on: May 21, 2014
 *      Author: richard
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
		epochAtLastUpdate_(0),
		firstTrainableLayerIndex_(0),
		lastRecurrentLayerIndex_(0)
{}

GradientBasedTrainer::~GradientBasedTrainer() {
	if (regularizer_)
		delete regularizer_;
}

u32 GradientBasedTrainer::requiredStatistics() {
	u32 requiredStatistics = estimator().requiredStatistics() | Statistics::objectiveFunctionStatistics;
	if (task_ == classification)
		requiredStatistics = requiredStatistics | Statistics::classificationStatistics;
	return requiredStatistics;
}

void GradientBasedTrainer::initialize() {
	Precursor::initialize();
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
	statistics_ = new Statistics((Statistics::StatisticTypes)requiredStatistics());
	statistics().initialize(network());
	regularizer_ = Regularizer::createRegularizer();
	criterion_ = TrainingCriterion::createCriterion();
	criterion_->initialize(network());
	lastRecurrentLayerIndex_ = std::min(network().lastRecurrentLayerIndex(), network().nLayer() - 1);
}

void GradientBasedTrainer::backpropTimeframe(u32 t, u32 layerIndexFrom, u32 layerIndexTo, bool greedyBackprop) {
	layerIndexTo = std::min(layerIndexTo, network().nLayer() - 1);
	u32 T = network().outputLayer().nTimeframes() - 1;
	/* hide all error signals of sequences that begin in time frame t+1 */
	if (t < T) {
		for (u32 l = layerIndexFrom; l <= layerIndexTo; l++) {
			if (network().layer(l).isInTrainingMode()) {
				network().layer(l).setErrorSignalVisibility(t+1, network().layer(l).activations(t,0).nColumns());
				network().layer(l).setErrorSignalVisibility(t, network().layer(l).activations(t,0).nColumns());
			}
		}
	}
	/* backpropagation through the layers */
	s32 L = layerIndexTo;
	// if t < T and greedy backprop: backpropagation only from last recurrent layer on
	// greedy backprop can only be used with sequence to single target mapping
	if ((t < T) && greedyBackprop)
		L = std::min(L, (s32)lastRecurrentLayerIndex_);
	for (s32 l = L; l >= (s32)std::max(layerIndexFrom, firstTrainableLayerIndex_); l--) {
		// exclude output layer at timeframe T because we need the initial error signal here
		if ((t < T) || (!network().layer(l).isOutputLayer())) {
			require_lt((u32)t, network().layer(l).nTimeframes());
			network().layer(l).backpropagate(t);
		}
	}
	/* undo the hiding */
	if (t < T) {
		for (u32 l = layerIndexFrom; l <= layerIndexTo; l++) {
			if (network().layer(l).isInTrainingMode()) {
				network().layer(l).setErrorSignalVisibility(t+1, network().layer(l).activations(t+1,0).nColumns());
			}
		}
	}
}

void GradientBasedTrainer::errorBackpropagation(u32 historyLength, bool greedyBackprop) {
	/* backpropagation through time */
	u32 T = network().outputLayer().nTimeframes() - 1;

	// backprop from the next layer that requires a full forward pass over all timeframes until the first layer is reached
	s32 layerIndexTo = network().nLayer() - 1;
	while (layerIndexTo >= 0) {
		s32 layerIndexFrom = layerIndexTo;
		// get layer up to which the forwarding can be done
		while ((layerIndexFrom > 0) && (!network().layer(layerIndexFrom).requiresFullPass()))
			layerIndexFrom--;
		// backprop all timeframes for the block [layerIndexFrom, layerIndexTo]
		for (s32 t = T; t > (s32)T - (s32)historyLength; t--) {
			backpropTimeframe((u32)t, (u32)layerIndexFrom, (u32)layerIndexTo, greedyBackprop);
		}
		// set the values for the next block
		layerIndexTo = layerIndexFrom - 1;
	}
}

void GradientBasedTrainer::updateGradient(u32 historyLength) {
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
					if(network().connection(c).type() == Connection::convolutionalConnection
							|| network().connection(c).type() == Connection::validConvolutionalConnection) {
#ifdef MODULE_CUDNN
						((ConvolutionalConnection&)network().connection(c)).backwardWRTKernel(statistics().weightsGradient(network().connection(c).name()),
								network().connection(c).from().activations(t, sourcePort), network().connection(c).to().errorSignal(t, destPort));
#else
						Matrix &activation = network().connection(c).from().activations(t, sourcePort);
						Matrix &errorSignal = network().connection(c).to().errorSignal(t, destPort);

						Matrix temp;
						((ConvolutionalConnection&)network().connection(c)).backwardPreprocess(errorSignal, temp);

						Matrix temp1;
						((ConvolutionalConnection&)network().connection(c)).forwardPreprocess(activation, temp1);

						statistics().weightsGradient(network().connection(c).name()).
								addMatrixProduct(temp1, temp, 1.0, 1.0, false, true);

						temp.finishComputation(false);
						temp1.finishComputation(false);
#endif
					}
					// in case of full weight matrices
					else {
						statistics().weightsGradient(network().connection(c).name()).addMatrixProduct(
								network().connection(c).from().activations(t, sourcePort),
								network().connection(c).to().errorSignal(t, destPort),
								1.0, 1.0, false, true);
					}
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
					if (network().layer(l).layerType() == Layer::batchNormalizationLayer) {
						Vector biasGradient;
						((BatchNormalizationLayer&)network().layer(l)).getBiasGradient(biasGradient);
						statistics().biasGradient(network().layer(l).name(), 0).add(biasGradient);
					}
					else if (network().layer(l).errorSignal(t, port).nRows() > 0) {
						statistics().biasGradient(network().layer(l).name(), port).addSummedColumnsChannelWise(
								network().layer(l).errorSignal(t, port), network().layer(l).nChannels(port));
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
	if (task_ == classification)
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

void FeedForwardTrainer::initialize() {
	Precursor::initialize();
	if (network_.isRecurrent())
		network().setMaximalMemory(2); // memorize the current and preceeding frame for correct forwarding of recurrencies
	else
		network().setMaximalMemory(1); // set the history length (number of time frames that are memorized) to 1
}

void FeedForwardTrainer::processBatch(Matrix& source, Matrix& targets) {
	require_gt(epochLength_, 0);
	require_eq(source.nColumns(), targets.nColumns());
	/* initial steps for the batch */
	// ensure everything is in computing state
	targets.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* forward the network */
	network().forward(source);

	/* compute initial error signal */
	computeInitialErrorSignal(targets);

	/* error backpropagation */
	errorBackpropagation();

	/* update statistics */
	Float objFctn = computeObjectiveFunction(targets);
	statistics().addToObjectiveFunction(objFctn);
	if (task_ == classification)
		statistics().increaseNumberOfClassificationErrors(nClassificationErrors(targets));
	statistics().increaseNumberOfObservations(source.nColumns());
	updateGradient();

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}

void FeedForwardTrainer::processSequenceBatch(MatrixContainer& source, MatrixContainer& targets) {
	require_gt(epochLength_, 0);
	for (u32 t = 0; t < source.nTimeframes(); t++) {
		require_eq(source.at(t).nColumns(), targets.at(t).nColumns());
	}
	/* initial steps for the batch */
	// ensure everything is in computing state
	targets.initComputation();
	network().initComputation();
	statistics().initComputation();

	network().reset();

	/* process each timeframe */
	for (u32 t = 0; t < source.nTimeframes(); t++) {
		network().forwardTimeframe(source, t, 0, network().nLayer() - 1, false);
		computeInitialErrorSignal(targets.at(t));
		errorBackpropagation(1, false);
		/* update statistics */
		Float objFctn = computeObjectiveFunction(targets.at(t));
		statistics().addToObjectiveFunction(objFctn);
		if (task_ == classification)
			statistics().increaseNumberOfClassificationErrors(nClassificationErrors(targets.at(t)));
		statistics().increaseNumberOfObservations(source.at(t).nColumns());
		updateGradient();
	}

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}

/*
 * RnnTrainer
 */
const Core::ParameterInt RnnTrainer::paramMaxTimeHistory_("max-time-history", Types::max<u32>(), "trainer");

const Core::ParameterBool RnnTrainer::paramGreedyForwarding_("greedy-forwarding", true, "trainer");

RnnTrainer::RnnTrainer() :
		Precursor(),
		maxTimeHistory_(Core::Configuration::config(paramMaxTimeHistory_)),
		greedyForwarding_(Core::Configuration::config(paramGreedyForwarding_))
{}

u32 RnnTrainer::requiredStatistics() {
	return (Precursor::requiredStatistics() | Statistics::sequenceCount);
}

void RnnTrainer::initialize() {
	require(maxTimeHistory_ > 0);
	Precursor::initialize();
}

void RnnTrainer::processSequenceBatch(MatrixContainer& source, Matrix& targets) {
	require_gt(epochLength_, 0);
	require_eq(source.getLast().nColumns(), targets.nColumns());

	/* initial steps for the sequence batch */
	// set the history length (number of time frames that are memorized)
	u32 historyLength = (maxTimeHistory_ < source.nTimeframes() ? maxTimeHistory_ : source.nTimeframes());
	network().setMaximalMemory(historyLength);
	// ensure everything is in computing state
	targets.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* forward the network */
	network().forwardSequence(source, greedyForwarding_);

	/* compute initial error signal */
	computeInitialErrorSignal(targets);

	/* error backpropagation */
	errorBackpropagation(historyLength);

	/* update statistics */
	Float objFctn = computeObjectiveFunction(targets);
	statistics().addToObjectiveFunction(objFctn);
	if (task_ == classification)
		statistics().increaseNumberOfClassificationErrors(nClassificationErrors(targets));
	statistics().increaseNumberOfObservations(source.getLast().nColumns());
	updateGradient(historyLength);

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}

void RnnTrainer::processSequenceBatch(MatrixContainer& source, MatrixContainer& targets) {
	require_gt(epochLength_, 0);
	for (u32 t = 0; t < source.nTimeframes(); t++) {
		require_eq(source.at(t).nColumns(), targets.at(t).nColumns());
	}

	/* initial steps for the sequence batch */
	// set the history length (number of time frames that are memorized)
	u32 historyLength = (maxTimeHistory_ < source.nTimeframes() ? maxTimeHistory_ : source.nTimeframes());
	network().setMaximalMemory(historyLength);
	// ensure everything is in computing state
	targets.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* forward the network without greedy forwarding */
	network().forwardSequence(source, false);

	/* compute initial error signal */
	computeInitialErrorSignals(targets);

	/* error backpropagation */
	errorBackpropagation(historyLength, false);

	/* update statistics */
	Float objFctn = computeObjectiveFunction(targets);
	statistics().addToObjectiveFunction(objFctn);
	if (task_ == classification)
		statistics().increaseNumberOfClassificationErrors(nClassificationErrors(targets));
	for (u32 t = 0; t < targets.nTimeframes(); t++)
		statistics().increaseNumberOfObservations(targets.at(t).nColumns());
	statistics().increaseNumberOfSequences(targets.getLast().nColumns());
	updateGradient(historyLength);

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nSequences())) {
		estimateModelParameters();
	}
}

/*
 * BagOfWordsNetworkTrainer
 */
BagOfWordsNetworkTrainer::BagOfWordsNetworkTrainer() :
		Precursor(),
		recurrentLayerIndex_(0)
{}

void BagOfWordsNetworkTrainer::initialize() {
	Precursor::initialize();
	// find index of recurrent layer
	for (u32 l = 0; l < network().nLayer(); l++) {
		if (network().layer(l).isRecurrent())
			recurrentLayerIndex_ = l;
	}
	recurrentErrorSignal_.resize(network().layer(recurrentLayerIndex_).nInputPorts());
	for (u32 port = 0; port < network().layer(recurrentLayerIndex_).nInputPorts(); port++)
		recurrentErrorSignal_.at(port).initComputation();
}

void BagOfWordsNetworkTrainer::framewiseErrorBackpropagation() {
	/* backpropagation through the layers */
	for (s32 l = (s32)recurrentLayerIndex_ - 1; l >= (s32)firstTrainableLayerIndex_; l--) {
		network().layer(l).backpropagate();
	}
}

void BagOfWordsNetworkTrainer::_updateGradient() {
	for (u32 l = 0; l <= recurrentLayerIndex_; l++) {
		for (u32 port = 0; port < network().layer(l).nInputPorts(); port++) {
			// update the bias gradients
			if (network().layer(l).useBias() && network().layer(l).isBiasTrainable()) {
				statistics().biasGradient(network().layer(l).name(), port).addSummedColumns(
						network().layer(l).latestErrorSignal(port));
			}
			// update the weights gradients
			for (u32 c = 0; c < network().layer(l).nIncomingConnections(port); c++) {
				if (network().layer(l).isTrainable(c, port)) {
					u32 sourcePort = network().layer(l).incomingConnection(c, port).sourcePort();
					u32 destPort = network().layer(l).incomingConnection(c, port).destinationPort();
					if(network().connection(c).type() == Connection::convolutionalConnection
							|| network().connection(c).type() == Connection::validConvolutionalConnection) {
#ifdef MODULE_CUDNN
						((ConvolutionalConnection&)network().connection(c)).backwardWRTKernel(statistics().weightsGradient(network().connection(c).name()),
								network().connection(c).from().latestActivations(sourcePort), network().connection(c).to().latestErrorSignal(destPort));
#else
						Matrix &activation = network().connection(c).from().activations(t, sourcePort);
						Matrix &errorSignal = network().connection(c).to().errorSignal(t, destPort);

						Matrix temp;
						((ConvolutionalConnection&)network().connection(c)).backwardPreprocess(errorSignal, temp);

						Matrix temp1;
						((ConvolutionalConnection&)network().connection(c)).forwardPreprocess(activation, temp1);

						statistics().weightsGradient(network().connection(c).name()).
								addMatrixProduct(temp1, temp, 1.0, 1.0, false, true);

						temp.finishComputation(false);
						temp1.finishComputation(false);
#endif
					}
					// in case of full weight matrices
					else {
						statistics().weightsGradient(network().layer(l).incomingConnection(c, port).name()).addMatrixProduct(
								network().layer(l).incomingConnection(c, port).from().latestActivations(sourcePort),
								network().layer(l).incomingConnection(c, port).to().latestErrorSignal(destPort),
								1.0, 1.0, false, true);
					}
				}
			}
		}
	}
}

void BagOfWordsNetworkTrainer::processSequenceBatch(MatrixContainer& source, Matrix& targets) {
	require_gt(epochLength_, 0);
	require_eq(source.getLast().nColumns(), targets.nColumns());

	/* initial steps for the sequence batch */
	// set the history length (number of time frames that are memorized)
	network().setMaximalMemory(2);
	// ensure everything is in computing state
	targets.initComputation();
	network().initComputation();
	statistics().initComputation();

	/* compute error signal of the sequence-length-normalization layer */
	// forward the sequence
	network().forwardSequence(source);
	// compute initial error signal
	computeInitialErrorSignal(targets);
	// error backpropagation
	errorBackpropagation();
	// update gradient for time frame T
	updateGradient(1);
	statistics().addToObjectiveFunction(computeObjectiveFunction(targets));
	if (task_ == classification)
		statistics().increaseNumberOfClassificationErrors(nClassificationErrors(targets));
	statistics().increaseNumberOfObservations(source.getLast().nColumns());

	// if any layer before recurrent layer index is trainable, proceed with these layers
	if (firstTrainableLayerIndex_ <= recurrentLayerIndex_) {
		// store error signal of recurrent layer
		for (u32 port = 0; port < network().layer(recurrentLayerIndex_).nInputPorts(); port++) {
			recurrentErrorSignal_.at(port).copyStructure(network().layer(recurrentLayerIndex_).latestErrorSignal(port));
			recurrentErrorSignal_.at(port).copy(network().layer(recurrentLayerIndex_).latestErrorSignal(port));
		}
		/* update the gradient for all remaining time frames */
		network().reset();
		network().setMaximalMemory(1);
		for (u32 t = 0; t < source.nTimeframes() - 1; t++) {
			// forward input up to recurrentLayerIndex_ - 1
			network().forward(source.at(t), 0, recurrentLayerIndex_ - 1);
			network().layer(recurrentLayerIndex_).addTimeframe(source.at(t).nColumns());
			// set the error signal for the recurrent layer
			for (u32 port = 0; port < network().layer(recurrentLayerIndex_).nInputPorts(); port++) {
				u32 nCols = network().layer(recurrentLayerIndex_).latestErrorSignal(port).nColumns();
				network().layer(recurrentLayerIndex_).latestErrorSignal(port).resize(
						recurrentErrorSignal_.at(port).nRows(), recurrentErrorSignal_.at(port).nColumns());
				network().layer(recurrentLayerIndex_).latestErrorSignal(port).copy(recurrentErrorSignal_.at(port));
				network().layer(recurrentLayerIndex_).latestErrorSignal(port).setVisibleColumns(nCols);
			 }
			// compute the remaining error signals and update the gradient
			framewiseErrorBackpropagation();
			_updateGradient();
		}
	}

	/* estimate the model parameters */
	if ((modelUpdateStrategy_ == afterBatch) || (epochLength_ <= statistics().nObservations())) {
		estimateModelParameters();
	}
}

/*
 * Special RNN Trainer
 */
const Core::ParameterInt SpecialRnnTrainer::paramEffectiveBatchSize_("effective-batch-size", 1, "trainer");

const Core::ParameterInt SpecialRnnTrainer::paramUpdateAfter_("update-after", 128, "trainer");

SpecialRnnTrainer::SpecialRnnTrainer() :
		Precursor(),
		effectiveBatchSize_(Core::Configuration::config(paramEffectiveBatchSize_)),
		currentFrameIdx_(0),
		currentSequenceIdx_(0)
{
	epochLength_ = Core::Configuration::config(paramUpdateAfter_);
}

void SpecialRnnTrainer::setEffectiveContainerSize(u32 sourceDim, u32 targetDim, u32 batchSize) {
	for (u32 t = 0; t < batchedSource_.nTimeframes(); t++) {
		batchedSource_.at(t).resize(sourceDim, batchSize);
	}
	batchedTargets_.resize(targetDim, batchSize);
}

void SpecialRnnTrainer::processSequenceBatch(MatrixContainer& source, Matrix& targets) {
	u32 historyLength = (maxTimeHistory_ < source.nTimeframes() ? maxTimeHistory_ : source.nTimeframes());

	// set the size for the new source and target containers
	batchedSource_.initComputation();
	batchedSource_.setMaximalMemory(historyLength);
	batchedSource_.reset();
	for (u32 t = 0; t < historyLength; t++) {
		batchedSource_.addTimeframe(source.getLast().nRows(), effectiveBatchSize_);
	}
	batchedTargets_.initComputation();
	batchedTargets_.resize(targets.nRows(), effectiveBatchSize_);

	targets.initComputation();
	source.initComputation();

	// fill the containers
	currentFrameIdx_ = 0;
	currentSequenceIdx_ = 0;
	u32 batchIdx = 0;
	u32 nRows = source.getLast().nRows();
	while ( (currentFrameIdx_ < source.nTimeframes()) && (currentSequenceIdx_ < source.getLast().nColumns()) ) {
		batchIdx = 0;
		batchedSource_.setToZero();
		/* create a batch from the sequences */
		while (batchIdx < effectiveBatchSize_) {
			// each sequence consists of historyLength frames
			for (u32 t = 0; t < historyLength; t++) {
				s32 frame = (s32)currentFrameIdx_ - (s32)t;
				bool frameExists = (frame >= 0) && (source.at((u32)frame).nColumns() > currentSequenceIdx_);
				if (frameExists) {
					batchedSource_.at(historyLength - t - 1).copyBlockFromMatrix(
							source.at((u32)frame), 0, currentSequenceIdx_, 0, batchIdx, nRows, 1);
				}
			}
			batchedTargets_.copyBlockFromMatrix(targets, 0, currentSequenceIdx_, 0, batchIdx, targets.nRows(), 1);
			// go to the next sequence or (if all sequences at this frame have been batched) to the next frame
			batchIdx++;
			currentSequenceIdx_++;
			if (currentSequenceIdx_ >= source.at(currentFrameIdx_).nColumns()) {
				currentSequenceIdx_ = 0;
				currentFrameIdx_++;
			}
			// abort criterion
			if (! ((currentFrameIdx_ < source.nTimeframes()) && (currentSequenceIdx_ < source.getLast().nColumns()) )) {
				break;
			}
		}
		// train on the batches
		setEffectiveContainerSize(source.getLast().nRows(), targets.nRows(), batchIdx);
		Precursor::processSequenceBatch(batchedSource_, batchedTargets_);
	}
}

void SpecialRnnTrainer::processSequenceBatch(MatrixContainer& source, MatrixContainer& targets) {
	Trainer::processSequenceBatch(source, targets);
}
