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
 * Trainer.cc
 *
 *  Created on: May 15, 2014
 *      Author: richard
 */

#include "Trainer.hh"
#include "Forwarder.hh"
#include "GradientBasedTrainer.hh"
#include <sstream>
#include <iomanip>

using namespace Nn;

/*
 * Trainer
 */
const Core::ParameterEnum Trainer::paramTrainer_("trainer",
		"none, feed-forward-trainer, rnn-trainer, bag-of-words-network-trainer, special-rnn-trainer",
		"none", "");

const Core::ParameterEnum Trainer::paramTask_("task", "classification, regression", "classification", "trainer");

// number of epochs to train for
const Core::ParameterInt Trainer::paramNumberOfEpochs_("number-of-epochs", 1, "trainer");

// epoch to start with (may be set to a higher value if training is continued at a later time)
const Core::ParameterInt Trainer::paramFirstEpoch_("start-with-epoch", 1, "trainer");

// define how many observations define an epoch (0 = complete dataset)
const Core::ParameterInt Trainer::paramEpochLength_("epoch-length", 0, "trainer");

// save after every n-th epoch (if 0 save only at the end of the training)
const Core::ParameterInt Trainer::paramSaveFrequency_("save-frequency", 1, "trainer");

Trainer::Trainer() :
		task_((TrainingTask)Core::Configuration::config(paramTask_)),
		numberOfEpochs_(Core::Configuration::config(paramNumberOfEpochs_)),
		nProcessedEpochs_(Core::Configuration::config(paramFirstEpoch_) - 1),
		epoch_(0),
		epochLength_(Core::Configuration::config(paramEpochLength_)),
		saveFrequency_(Core::Configuration::config(paramSaveFrequency_)),
		nProcessedMinibatches_(0),
		nProcessedObservations_(0),
		minibatchGenerator_(supervised),
		estimator_(0),
		isInitialized_(false)
{
	// first epoch must be at least 1
	require_ge(Core::Configuration::config(paramFirstEpoch_), 1);
}

Trainer::~Trainer() {
	if (estimator_)
		delete estimator_;
}

void Trainer::initialize() {
	if (!isInitialized_) {
		// initialize the network with a maximal memory = 1 for activations/error signals over time
		// (only activations/error signals of most recent time frame are stored)
		network_.initialize();
		minibatchGenerator_.initialize();
		estimator_ = Estimator::createEstimator();
		estimator_->initialize(network_, task_);
		if (epochLength_ == 0)
			epochLength_ = minibatchGenerator_.totalNumberOfObservations();
		isInitialized_ = true;
	}
}

void Trainer::finalize() {
	require(estimator_);
	estimator_->finalize();
}

void Trainer::processAllEpochs(u32 batchSize) {
	Core::Log::openTag("neural-network.process-all-epochs");
	while (nProcessedEpochs_ < numberOfEpochs_) {
		Core::Log::os("Start epoch ") << nProcessedEpochs_ + 1;
		estimator_->setEpoch(nProcessedEpochs_ + 1);
		processEpoch(batchSize);
		nProcessedEpochs_++;
		nProcessedObservations_ = 0;
		nProcessedMinibatches_ = 0;
	}
	if (saveFrequency_ == 0) { // save at least at the end of the training
		network().saveNeuralNetworkParameters();
	}
	Core::Log::closeTag();
}

void Trainer::processEpoch(u32 batchSize) {
	Core::Log::openTag("neural-network.process-epoch");
	while (nProcessedObservations_ < epochLength_)
		processBatch(batchSize);
	Core::Log::os("Processed ") << nProcessedObservations_ << " observations in " << nProcessedMinibatches_ << " mini-batches.";
	if ( (saveFrequency_ > 0) && ((nProcessedEpochs_ + 1) % saveFrequency_ == 0) ) {
		std::stringstream s;
		s << ".epoch-" << nProcessedEpochs_ + 1;
		network().saveNeuralNetworkParameters(s.str());
	}
	Core::Log::closeTag();
}

void Trainer::processBatch(u32 batchSize) {
	Core::Log::openTag("neural-network.process-batch");
	// ensure that last batch in epoch processes only the remaining observations
	if (nProcessedObservations_ + batchSize > epochLength_)
		batchSize = epochLength_ - nProcessedObservations_;
	minibatchGenerator_.generateBatch(batchSize);
	Core::Log::os() << "Process mini-batch " << nProcessedMinibatches_ + 1 << " with " << batchSize << " observations.";

	/* call the trainer */
	if (minibatchGenerator_.sourceType() == single)
		processBatch(minibatchGenerator_.sourceBatch(), minibatchGenerator_.targetBatch());
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == single))
		processSequenceBatch(minibatchGenerator_.sourceSequenceBatch(), minibatchGenerator_.targetBatch());
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == sequence))
		processSequenceBatch(minibatchGenerator_.sourceSequenceBatch(), minibatchGenerator_.targetSequenceBatch());

	nProcessedMinibatches_++;
	nProcessedObservations_ += batchSize;
	Core::Log::closeTag();
}

NeuralNetwork& Trainer::network() {
	require(isInitialized_);
	return network_;
}

Estimator& Trainer::estimator() {
	require(estimator_);
	return *estimator_;
}

void Trainer::processBatch(Matrix& source, Matrix& target) {
	Core::Error::msg("Trainer::processBatch(source, target): Supervised frame-wise training not supported by this trainer.") << Core::Error::abort;
}

void Trainer::processSequenceBatch(MatrixContainer& source, Matrix& target) {
	Core::Error::msg("Trainer::processBatch(source, target): Supervised sequence training with one target per sequence "
			"is not supported by this trainer.") << Core::Error::abort;
}

void Trainer::processSequenceBatch(MatrixContainer& source, MatrixContainer& target) {
	Core::Error::msg("Trainer::processBatch(source, target): Supervised sequence training with framewise sequence targets "
			"is not supported by this trainer.") << Core::Error::abort;
}

/* factory */
Trainer* Trainer::createTrainer() {
	Trainer* trainer = 0;

	switch ( (TrainerType) Core::Configuration::config(paramTrainer_) ) {
	case feedForwardTrainer:
		Core::Log::os("Create feed-forward-trainer.");
		trainer = new FeedForwardTrainer();
		break;
	case rnnTrainer:
		Core::Log::os("Create rnn-trainer.");
		trainer = new RnnTrainer();
		break;
	case bagOfWordsNetworkTrainer:
		Core::Log::os("Create bag-of-words-network-trainer.");
		trainer = new BagOfWordsNetworkTrainer();
		break;
	case specialRnnTrainer:
		Core::Log::os("Create special-rnn-trainer.");
		trainer = new SpecialRnnTrainer();
		break;
	default:
		Core::Error::msg("Trainer::createTrainer: No trainer chosen.") << Core::Error::abort;
		break;
	}

	return trainer;
}
