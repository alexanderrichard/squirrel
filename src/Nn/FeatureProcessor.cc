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

#include "FeatureProcessor.hh"
#include <Math/Random.hh>

using namespace Nn;
using namespace std;

/*
 * BaseFeatureProcessor
 */

const Core::ParameterEnum BaseFeatureProcessor::paramTrainingMode_("training-mode",
		"supervised, unsupervised", "supervised", "");

const Core::ParameterInt BaseFeatureProcessor::paramBatchSize_("batch-size", 1, "");

const Core::ParameterInt BaseFeatureProcessor::paramMaxNumberOfEpochs_("max-number-of-epochs", 1, "");

// epoch to start with (may be set to a higher value if training is continued at a later time)
const Core::ParameterInt BaseFeatureProcessor::paramFirstEpoch_("start-with-epoch", 1, "");

// define how many observations define an epoch (0 = complete dataset)
const Core::ParameterInt BaseFeatureProcessor::paramEpochLength_("epoch-length", 0, "");

const Core::ParameterEnum BaseFeatureProcessor::paramFeatureType_("feature-type", "single, sequence", "single", "");

// save after every n-th epoch (if 0 save only at the end of the training)
const Core::ParameterInt BaseFeatureProcessor::paramSaveFrequency_("save-frequency", 1, "");

BaseFeatureProcessor::BaseFeatureProcessor() :
		batchSize_(Core::Configuration::config(paramBatchSize_)),
		maxNumberOfEpochs_(Core::Configuration::config(paramMaxNumberOfEpochs_)),
		nProcessedEpochs_(Core::Configuration::config(paramFirstEpoch_) - 1),
		epochLength_(Core::Configuration::config(paramEpochLength_)),
		nProcessedMinibatches_(0),
		nProcessedObservations_(0),
		dimension_(0),
		totalNumberOfObservations_(0),
		saveFrequency_(Core::Configuration::config(paramSaveFrequency_)),
		trainingMode_((TrainingMode)Core::Configuration::config(paramTrainingMode_)),
		trainer_(0),
		isInitialized_(false)
{
	// first epoch must be at least 1
	require_ge(Core::Configuration::config(paramFirstEpoch_), 1);
	// batch size must be at least 1
	require_ge(batchSize_, 1);
}

BaseFeatureProcessor::~BaseFeatureProcessor() {
	if (trainer_)
		delete trainer_;
}

void BaseFeatureProcessor::finalize() {
	if (trainer_)
		trainer_->finalize();
}

void BaseFeatureProcessor::processEpoch() {
	Core::Log::openTag("neural-network.process-epoch");
	startNewEpoch();
	while (nProcessedObservations_ < epochLength_) {
		require(trainer_);
		trainer_->setEpoch(nProcessedEpochs_ + 1);
		processBatch();
	}
	Core::Log::os("Processed ") << nProcessedObservations_ << " observations in " << nProcessedMinibatches_ << " mini-batches";
	if ( (saveFrequency_ > 0) && ((nProcessedEpochs_ + 1) % saveFrequency_ == 0) ) {
		std::stringstream s;
		s << ".epoch-" << nProcessedEpochs_ + 1;
		trainer_->network().saveNeuralNetworkParameters(s.str());
	}
	Core::Log::closeTag();
}

void BaseFeatureProcessor::processAllEpochs() {
	Core::Log::openTag("neural-network.process-all-epochs");
	while (nProcessedEpochs_ < maxNumberOfEpochs_) {
		Core::Log::os("Start epoch ") << nProcessedEpochs_ + 1;
		processEpoch();
		nProcessedEpochs_++;
		nProcessedObservations_ = 0;
		nProcessedMinibatches_ = 0;
	}
	if (saveFrequency_ == 0) { // save at least at the end of the training
		trainer_->network().saveNeuralNetworkParameters();
	}
	Core::Log::closeTag();
}

/* factory */
BaseFeatureProcessor* BaseFeatureProcessor::createFeatureProcessor() {
	BaseFeatureProcessor* featureProcessor = 0;
	switch ( (FeatureType) Core::Configuration::config(paramFeatureType_) ) {
	case single:
		featureProcessor = new FeatureProcessor();
		break;
	case sequence:
		featureProcessor = new SequenceFeatureProcessor();
		break;
	default:
		break;
	}
	return featureProcessor;
}

/*
 * FeatureProcessor
 */
FeatureProcessor::FeatureProcessor() :
		Precursor()
{}

void FeatureProcessor::startNewEpoch() {
	require(isInitialized_);
	if (trainingMode_ == supervised)
		labeledFeatureReader_.newEpoch();
	else // if trainingMode_ == unsupervised
		featureReader_.newEpoch();
}

void FeatureProcessor::generateMinibatch() {
	batch_.finishComputation(false);
	labels_.finishComputation(false);
	u32 remainingObservations = epochLength_ - nProcessedObservations_;
	u32 nObservations = std::min(batchSize_, remainingObservations);
	batch_.resize(dimension_, nObservations);
	labels_.resize(nObservations);
	// copy data into mini-batch
	for (u32 j = 0; j < nObservations; j++) {
		if (!hasUnprocessedFeatures()) {
			startNewEpoch();
		}
		const Math::Vector<Float>& f = (trainingMode_ == supervised ?
				labeledFeatureReader_.next() : featureReader_.next());
		u32 label = (trainingMode_ == supervised ? labeledFeatureReader_.label() : 0);
		for (u32 i = 0; i < dimension_; i++) {
			batch_.at(i,j) = f.at(i);
		}
		// add label
		labels_.at(j) = label;
	}
	nProcessedObservations_ += nObservations;
}

void FeatureProcessor::initialize() {
	// initialize trainer
	trainer_ = Trainer::createFramewiseTrainer();
	if (trainingMode_ == supervised) {
		require(trainer_->isSupervised());
	}
	else { // if trainingMode_ == unsupervised
		require(trainer_->isUnsupervised());
	}
	trainer_->initialize(epochLength_);
	// initialize feature reader
	if (trainingMode_ == supervised)
		labeledFeatureReader_.initialize();
	else // if trainingMode_ == unsupervised
		featureReader_.initialize();

	dimension_ = (trainingMode_ == supervised ?
			labeledFeatureReader_.featureDimension() : featureReader_.featureDimension());
	totalNumberOfObservations_ = (trainingMode_ == supervised ?
			labeledFeatureReader_.totalNumberOfFeatures() : featureReader_.totalNumberOfFeatures());

	if (epochLength_ == 0) {
		epochLength_ = totalNumberOfObservations_;
	}

	isInitialized_ = true;
}

bool FeatureProcessor::hasUnprocessedFeatures() const {
	require(isInitialized_);
	return (trainingMode_ == supervised ?
			labeledFeatureReader_.hasFeatures() : featureReader_.hasFeatures());
}

void FeatureProcessor::processBatch() {
	require(isInitialized_);
	require(trainer_);
	Core::Log::openTag("neural-network.process-batch");
	generateMinibatch();
	if (batch_.nColumns() < batchSize_) {
		Core::Log::os() << "Only " << batch_.nColumns() << " observations left in buffer. Set batch size to " << batch_.nColumns() << ".";
	}
	Core::Log::os() << "Process mini-batch " << nProcessedMinibatches_ + 1 << " with " << batch_.nColumns() << " observations.";
	/* call the trainer (depending on trainingMode_) */
	if (trainingMode_ == supervised)
		trainer_->processBatch(batch_, labels_);
	else
		trainer_->processBatch(batch_);
	nProcessedMinibatches_++;
	Core::Log::closeTag();
}

/*
 * SequenceFeatureProcessor
 */
SequenceFeatureProcessor::SequenceFeatureProcessor() :
		Precursor()
{}

void SequenceFeatureProcessor::startNewEpoch() {
	require(isInitialized_);
	if (trainingMode_ == supervised)
		labeledFeatureReader_.newEpoch();
	else // if trainingMode_ == unsupervised
		featureReader_.newEpoch();
}

void SequenceFeatureProcessor::generateMinibatch() {
	require(isInitialized_);
	batch_.reset();
	batch_.finishComputation(false);
	labels_.finishComputation(false);
	// if mini-batch size is larger than one, the sequences should be sorted in descending order according to their length
	u32 bufferSize = (trainingMode_ == supervised ? labeledFeatureReader_.maxBufferSize() : featureReader_.maxBufferSize());
	if ((bufferSize % batchSize_ != 0) && (bufferSize < totalNumberOfObservations_)) {
		std::cerr << "SequenceFeatureProcessor: buffer-size has to be a multiple of the batch size or at least "
				<< totalNumberOfObservations_ << ". Abort." << std::endl;
		exit(1);
	}
	if ((epochLength_ % batchSize_ != 0) && (epochLength_ != totalNumberOfObservations_)) {
		std::cerr << "SequenceFeatureProcessor: epoch-length has to be a multiple of the batch size or "
				<< totalNumberOfObservations_ << ". Abort." << std::endl;
		exit(1);
	}
	bool isSorted = (trainingMode_ == supervised ? labeledFeatureReader_.areSequencesSorted() : featureReader_.areSequencesSorted());
	if ((batchSize_ > 1) && (!isSorted)) {
		std::cerr << "SequenceFeatureProcessor: mini-batching only possible with sorted sequences. Abort." << std::endl;
		exit(1);
	}
	if (!hasUnprocessedFeatures()) {
		startNewEpoch();
	}
	u32 remainingSequences = epochLength_ - nProcessedObservations_;
	u32 nSequences = std::min(batchSize_, remainingSequences);
	labels_.resize(nSequences);

	// local copy of all the sequences the mini-batch is build from
	std::vector< Math::Matrix<Float> > tmpBatch;
	std::vector<u32> sequenceLength(nSequences); // store the length of all sequences
	for (u32 sequenceIndex = 0; sequenceIndex < nSequences; sequenceIndex++) {
		const Math::Matrix<Float>& f = (trainingMode_ == supervised ?
				labeledFeatureReader_.next() : featureReader_.next());
		u32 label = (trainingMode_ == supervised ? labeledFeatureReader_.label() : 0);
		// add label
		labels_.at(sequenceIndex) = label;
		// copy sequence
		tmpBatch.push_back(Math::Matrix<Float>(f));
	}

	// how many sequences are active (started) at time frame t?
	u32 maxSequenceLength = tmpBatch.at(0).nColumns();
	std::vector<u32> nStartedSequences(maxSequenceLength, 0);
	u32 i = 0;
	for (u32 t = 0; t < maxSequenceLength; t++) {
		if (t > 0)
			nStartedSequences.at(t) = nStartedSequences.at(t - 1);
		while ( (i < nSequences) && (maxSequenceLength - tmpBatch.at(i).nColumns() == t) ) {
			nStartedSequences.at(t)++;
			i++;
		}
	}

	// add all time frames to the actual sequence batch (note that the sequences are ordered!)
	batch_.setMaximalMemory(maxSequenceLength);
	for (u32 t = 0; t < maxSequenceLength; t++) {
		batch_.addTimeframe(dimension_, nStartedSequences.at(t));
		for (u32 i = 0; i < nStartedSequences.at(t); i++) {
			for (u32 d = 0; d < dimension_; d++) {
				u32 offset = maxSequenceLength - tmpBatch.at(i).nColumns();
				batch_.getLast().at(d, i) = tmpBatch.at(i).at(d, t - offset);
			}
		}
	}

	nProcessedObservations_ += nSequences;
}

void SequenceFeatureProcessor::initialize() {
	// initialize trainer
	trainer_ = Trainer::createSequenceTrainer();
	if (trainingMode_ == supervised) {
		require(trainer_->isSupervised());
	}
	else { // if trainingMode_ == unsupervised
		require(trainer_->isUnsupervised());
	}
	trainer_->initialize(epochLength_);

	// initialize feature reader
	if (trainingMode_ == supervised)
		labeledFeatureReader_.initialize();
	else // if trainingMode_ == unsupervised
		featureReader_.initialize();

	dimension_ = (trainingMode_ == supervised ?
			labeledFeatureReader_.featureDimension() : featureReader_.featureDimension());
	totalNumberOfObservations_ = (trainingMode_ == supervised ?
			labeledFeatureReader_.totalNumberOfSequences() : featureReader_.totalNumberOfSequences());

	if (epochLength_ == 0) {
		epochLength_ = totalNumberOfObservations_;
	}

	isInitialized_ = true;
}

bool SequenceFeatureProcessor::hasUnprocessedFeatures() const {
	require(isInitialized_);
	return (trainingMode_ == supervised ?
			labeledFeatureReader_.hasSequences() : featureReader_.hasSequences());
}

void SequenceFeatureProcessor::processBatch() {
	require(trainer_);
	Core::Log::openTag("neural-network.process-batch");
	generateMinibatch();
	if (batch_.getLast().nColumns() < batchSize_) {
		Core::Log::os() << "Only " << batch_.getLast().nColumns() << " sequences left in buffer. Set batch size to "
				<< batch_.getLast().nColumns() << ".";
	}
	Core::Log::os() << "Process mini-batch " << nProcessedMinibatches_ + 1 << " with " << batch_.getLast().nColumns() << " observations.";
	/* call the trainer (depending on trainingMode_) */
	if (trainingMode_ == supervised)
		trainer_->processSequenceBatch(batch_, labels_);
	else
		trainer_->processSequenceBatch(batch_);
	nProcessedMinibatches_++;
	Core::Log::closeTag();
}
