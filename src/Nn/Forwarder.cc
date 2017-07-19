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
 * Forwarder.cc
 *
 *  Created on: Jan 18, 2017
 *      Author: richard
 */

#include "Forwarder.hh"

using namespace Nn;

/*
 * Forwarder
 */
const Core::ParameterEnum Forwarder::paramTask_("task", "evaluate, dump-output", "evaluate", "forwarder");

const Core::ParameterEnum Forwarder::paramEvaluate_("evaluate", "none, classification-error, cross-entropy, squared-error", "none", "forwarder");

Forwarder::Forwarder() :
		task_((Task)Core::Configuration::config(paramTask_)),
		evaluation_((Evaluation)Core::Configuration::config(paramEvaluate_)),
		minibatchGenerator_(task_ == evaluate ? supervised : unsupervised),
		writer_(0),
		evalResult_(0),
		evalNormalization_(0),
		isInitialized_(false)
{}

Forwarder::~Forwarder() {
	if (writer_)
		delete writer_;
}

void Forwarder::restoreSequenceOrder(const std::vector< Math::Matrix<Float> >& in, std::vector< Math::Matrix<Float> >& out) {
	out.clear();
	const std::vector<u32>& order = minibatchGenerator_.sequenceOrder();
	// compute original sequence lengths
	std::vector<u32> lengths(order.size());
	for (u32 t = 0; t < in.size(); t++) {
		for (u32 col = 0; col < in.at(t).nColumns(); col++)
			lengths.at(order.at(col))++;
	}
	// restore sequences in Math::Matrix format
	u32 dim = in.back().nRows();
	for (u32 i = 0; i < lengths.size(); i++)
		out.push_back(Math::Matrix<Float>(dim, lengths.at(i)));
	for (u32 t = 0; t < in.size(); t++) {
		for (u32 col = 0; col < in.at(t).nColumns(); col++) {
			u32 n = t - (in.size() - lengths.at(order.at(col)));
#pragma omp parallel for
			for (u32 d = 0; d < dim; d++)
				out.at(order.at(col)).at(d, n) = in.at(t).at(d, col);
		}
	}
}

void Forwarder::dumpBatch(Matrix& source) {
	require(isInitialized_);
	network_.forward(source);
	network_.outputLayer().finishComputation();
	Math::Matrix<Float> result(network_.outputDimension(), source.nColumns());
	result.copy(network_.outputLayer().latestActivations(0).begin());
	network_.outputLayer().initComputation(false);
	writer_->write(result);
}

void Forwarder::dumpBatch(MatrixContainer& source) {
	require(isInitialized_);
	u32 memory = (network_.isRecurrent() ? 2 : 1);
	memory = (network_.requiresFullMemorization() ? source.nTimeframes() : memory);
	network_.setMaximalMemory(memory);
	network_.forwardSequence(source, true);
	// restore sequence ordering
	network_.outputLayer().finishComputation();
	Math::Matrix<Float> result(network_.outputDimension(), source.getLast().nColumns());
	for (u32 col = 0; col < result.nColumns(); col++) {
#pragma omp parallel for
		for (u32 d = 0; d < result.nRows(); d++)
			result.at(d, minibatchGenerator_.sequenceOrder().at(col)) = network_.outputLayer().latestActivations(0).at(d, col);
	}
	network_.outputLayer().initComputation(false);
	writer_->write(result);
}

void Forwarder::dumpSequenceBatch(MatrixContainer& source) {
	require(isInitialized_);

	/* forward */
	std::vector< Math::Matrix<Float> > tmp;
	if (network_.requiresFullMemorization()) {
		network_.setMaximalMemory(source.nTimeframes());
		network_.forwardSequence(source, false);
		network_.outputLayer().finishComputation();
		for (u32 t = 0; t < source.nTimeframes(); t++) {
			tmp.push_back(Math::Matrix<Float>(network_.outputDimension(), network_.outputLayer().activations(t, 0).nColumns()));
			tmp.back().copy(network_.outputLayer().activations(t, 0).begin());
		}
		network_.outputLayer().initComputation(false);
	}
	else {
		network_.setMaximalMemory(network_.isRecurrent() ? 2 : 1);
		network_.reset();
		for (u32 t = 0; t < source.nTimeframes(); t++) {
			network_.forwardTimeframe(source, t, 0, network_.nLayer(), false);
			network_.outputLayer().finishComputation();
			tmp.push_back(Math::Matrix<Float>(network_.outputDimension(), network_.outputLayer().latestActivations(0).nColumns()));
			tmp.back().copy(network_.outputLayer().latestActivations(0).begin());
			network_.outputLayer().initComputation(false);
		}
	}

	/* restore correct sequence order */
	std::vector< Math::Matrix<Float> > out;
	restoreSequenceOrder(tmp, out);

	/* write result */
	for (u32 i = 0; i < out.size(); i++)
		writer_->write(out.at(i));
}

Float Forwarder::evaluateBatch(Matrix& source, Matrix& targets) {
	require(isInitialized_);
	network_.forward(source);
	targets.initComputation();
	switch (evaluation_) {
	case classificationError:
		return (Float)network_.outputLayer().latestActivations(0).nClassificationErrors(targets);
		break;
	case crossEntropy:
		return network_.outputLayer().latestActivations(0).crossEntropyObjectiveFunction(targets);
		break;
	case squaredError:
		return network_.outputLayer().latestActivations(0).squaredErrorObjectiveFunction(targets);
		break;
	default:
		break;
	}
	return 0;
}

Float Forwarder::evaluateSequenceBatch(MatrixContainer& source, Matrix& targets) {
	require(isInitialized_);
	u32 memory = (network_.isRecurrent() ? 2 : 1);
	memory = (network_.requiresFullMemorization() ? source.nTimeframes() : memory);
	network_.setMaximalMemory(memory);
	network_.forwardSequence(source, true);
	targets.initComputation();
	switch (evaluation_) {
	case classificationError:
		return (Float)network_.outputLayer().latestActivations(0).nClassificationErrors(targets);
		break;
	case crossEntropy:
		return network_.outputLayer().latestActivations(0).crossEntropyObjectiveFunction(targets);
		break;
	case squaredError:
		return network_.outputLayer().latestActivations(0).squaredErrorObjectiveFunction(targets);
		break;
	default:
		break;
	}
	return 0;
}

Float Forwarder::evaluateSequenceBatch(MatrixContainer& source, MatrixContainer& targets) {
	require(isInitialized_);
	// store all timeframes during forwarding
	network_.setMaximalMemory(source.nTimeframes());
	network_.forwardSequence(source, false);
	targets.initComputation();
	Float result = 0;
	for (u32 t = 0; t < source.nTimeframes(); t++) {
		switch (evaluation_) {
		case classificationError:
			result += (Float)network_.outputLayer().activations(t, 0).nClassificationErrors(targets.at(t));
			break;
		case crossEntropy:
			result += network_.outputLayer().activations(t, 0).crossEntropyObjectiveFunction(targets.at(t));
			break;
		case squaredError:
			result += network_.outputLayer().activations(t, 0).squaredErrorObjectiveFunction(targets.at(t));
			break;
		default:
			break;
		}
	}
	return result;
}

void Forwarder::dump() {
	require(isInitialized_);
	if (minibatchGenerator_.sourceType() == single) {
		dumpBatch(minibatchGenerator_.sourceBatch());
	}
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == single)) {
		dumpBatch(minibatchGenerator_.sourceSequenceBatch());
	}
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == sequence)) {
		dumpSequenceBatch(minibatchGenerator_.sourceSequenceBatch());
	}
}

void Forwarder::eval() {
	require(isInitialized_);
	if (minibatchGenerator_.sourceType() == single) {
		evalResult_ += evaluateBatch(minibatchGenerator_.sourceBatch(), minibatchGenerator_.targetBatch());
		evalNormalization_ += minibatchGenerator_.targetBatch().nColumns();
	}
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == single)) {
		evalResult_ += evaluateSequenceBatch(minibatchGenerator_.sourceSequenceBatch(), minibatchGenerator_.targetBatch());
		evalNormalization_ += minibatchGenerator_.targetBatch().nColumns();
	}
	else if ((minibatchGenerator_.sourceType() == sequence) && (minibatchGenerator_.targetType() == sequence)) {
		evalResult_ += evaluateSequenceBatch(minibatchGenerator_.sourceSequenceBatch(), minibatchGenerator_.targetSequenceBatch());
		for (u32 t = 0; t < minibatchGenerator_.targetSequenceBatch().nTimeframes(); t++)
			evalNormalization_ += minibatchGenerator_.targetSequenceBatch().at(t).nColumns();
	}
}

void Forwarder::initialize() {
	minibatchGenerator_.initialize();
	network_.initialize();
	network_.initComputation();
	/* initialize feature writer for feature dumping */
	if (task_ == dumpFeatures) {
		if (minibatchGenerator_.targetType() == single) {
			writer_ = new Features::FeatureWriter();
			writer_->initialize(minibatchGenerator_.totalNumberOfObservations(), network_.outputDimension());
		}
		else { // (minibatchGenerator_.targetType() == sequence)
			writer_ = new Features::SequenceFeatureWriter();
			dynamic_cast< Features::SequenceFeatureWriter* >(writer_)->initialize(minibatchGenerator_.totalNumberOfFeatures(),
					network_.outputDimension(), minibatchGenerator_.totalNumberOfObservations());
		}
	}
	isInitialized_ = true;
}

void Forwarder::forward(u32 batchSize) {
	require(isInitialized_);
	u32 nProcessedObservations = 0;
	while (nProcessedObservations < minibatchGenerator_.totalNumberOfObservations()) {
		u32 bsize = std::min(batchSize, minibatchGenerator_.totalNumberOfObservations() - nProcessedObservations);
		minibatchGenerator_.generateBatch(bsize);
		if (task_ == evaluate)
			eval();
		else
			dump();
		nProcessedObservations += bsize;
	}
}

void Forwarder::finalize() {
	if (task_ == evaluate) {
		Core::Log::os("Processed ") << evalNormalization_ << " features.";
		switch (evaluation_) {
		case classificationError:
			Core::Log::os("Classification error rate: ") << evalResult_ / evalNormalization_;
			break;
		case crossEntropy:
			Core::Log::os("Cross-entropy loss: ") << evalResult_ / evalNormalization_;
			break;
		case squaredError:
			Core::Log::os("Squared error loss: ") << evalResult_ / evalNormalization_;
			break;
		case none:
		default:
			break;
		}
	}
}
