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
 * FramewiseScorer.cc
 *
 *  Created on: May 12, 2017
 *      Author: richard
 */

#include "Scorer.hh"
#include <Nn/FeatureTransformation.hh>

using namespace Hmm;

/*
 * Scorer
 */
const Core::ParameterEnum Scorer::paramScorerType_("type", "framewise-neural-network-scorer, segment-scorer",
		"framewise-neural-network-scorer", "scorer");

Scorer::Scorer() :
		nClasses_(0),
		isInitialized_(false),
		sequence_(0)
{}

void Scorer::initialize() {
	isInitialized_ = true;
}

void Scorer::setSequence(const Math::Matrix<Float>& sequence) {
	require(isInitialized_);
	sequence_ = &sequence;
}

Scorer* Scorer::create() {
	switch ((ScorerType) Core::Configuration::config(paramScorerType_)) {
	case framewiseNeuralNetworkScorer:
		Core::Log::os("Create framewise-neural-network-scorer.");
		return new FramewiseNeuralNetworkScorer();
		break;
	case segmentScorer:
		Core::Log::os("Create segment-scorer.");
		return new SegmentScorer();
		break;
	default:
		return 0; // this can not happen
	}
}


/*
 * FramewiseNeuralNetworkScorer
 */
const Core::ParameterString FramewiseNeuralNetworkScorer::paramPriorFile_("prior-file", "", "scorer");

const Core::ParameterFloat FramewiseNeuralNetworkScorer::paramPriorScale_("prior-scale", 1.0, "scorer");

const Core::ParameterBool FramewiseNeuralNetworkScorer::paramLogarithmizeNetworkOutput_("logarithmize-network-output", true, "scorer");

const Core::ParameterInt FramewiseNeuralNetworkScorer::paramBatchSize_("batch-size", 1, "scorer");

FramewiseNeuralNetworkScorer::FramewiseNeuralNetworkScorer() :
		Precursor(),
		priorFile_(Core::Configuration::config(paramPriorFile_)),
		priorScale_(Core::Configuration::config(paramPriorScale_)),
		logarithmizeNetworkOutput_(Core::Configuration::config(paramLogarithmizeNetworkOutput_)),
		batchSize_(Core::Configuration::config(paramBatchSize_))
{}

void FramewiseNeuralNetworkScorer::initialize() {
	Precursor::initialize();
	network_.initialize();
	nClasses_ = network_.outputDimension();
	prior_.resize(nClasses_);
	// initialize prior
	if (!priorFile_.empty()) {
		prior_.read(priorFile_);
		Core::Log::os("Load prior from ") << priorFile_;
		if (prior_.size() != nClasses_)
			Core::Error::msg("FramewiseNeuralNetworkScorer::initialized: prior-file must be a ") << nClasses_ << " dimensional vector." << Core::Error::abort;
	}
	prior_.initComputation();
	if (priorFile_.empty())
		prior_.fill(0.0);
	else
		prior_.log();
}

void FramewiseNeuralNetworkScorer::setSequence(const Math::Matrix<Float>& sequence) {
	scores_.initComputation();
	scores_.resize(network_.outputDimension(), sequence.nColumns());

	// process input sequence in several batches
	for (u32 colIdx = 0; colIdx < sequence.nColumns(); colIdx += batchSize_) {

		u32 batchSize = std::min(batchSize_, sequence.nColumns() - colIdx);
		Nn::Matrix batch(sequence.nRows(), batchSize);
		batch.copyBlockFromMatrix(sequence, 0, colIdx, 0, 0, sequence.nRows(), batchSize);

		Nn::FeatureTransformation featureTransformation(Nn::single);

		// if the feature transformation creates sequences out of the frames...
		if (featureTransformation.outputFormat() == Nn::sequence) {
			Nn::MatrixContainer input;
			featureTransformation.transform(batch, input);
			network_.setMaximalMemory(2);
			network_.forwardSequence(input, true);
		}
		// ... else simple work with the frames
		else {
			network_.forward(batch);
		}

		// store result in scores
		scores_.copyBlockFromMatrix(network_.outputLayer().latestActivations(0), 0, 0, 0, colIdx, network_.outputDimension(), batchSize);
	}

	if (logarithmizeNetworkOutput_)
		scores_.log();
	// subtract log prior
	scores_.addToAllColumns(prior_, -priorScale_);
	// make sure scores are all negative
	scores_.addConstantElementwise(-scores_.maxValue());
	scores_.finishComputation();
}

Float FramewiseNeuralNetworkScorer::frameScore(u32 t, u32 c) {
	require(isInitialized_);
	require_lt(c, scores_.nRows());
	require_lt(t, scores_.nColumns());
	return scores_.at(c, t);
}


/*
 * SegmentScorer
 */
const Core::ParameterBool SegmentScorer::paramScaleSegmentByLength_("scale-segment-by-length", true, "scorer");

SegmentScorer::SegmentScorer() :
		Precursor(),
		scaleSegmentByLength_(Core::Configuration::config(paramScaleSegmentByLength_)),
		t_start_(Types::max<u32>()),
		t_end_(Types::max<u32>())
{}

void SegmentScorer::initialize() {
	Precursor::initialize();
	prior_.finishComputation();
	networkInput_.resize(network_.inputDimension(), batchSize_);
}

void SegmentScorer::generateVector(u32 t_start, u32 t_end, u32 column, Nn::Matrix& result) {
	require_le(t_start, t_end);
	require_eq(result.nRows(), sequence_.nRows());
	require_lt(column, result.nColumns());
#pragma omp parallel for
	for (u32 row = 0; row < sequence_.nRows(); row++) {
		result.at(row, column) = sequence_.at(row, t_end);
		if (t_start > 0) {
			result.at(row, column) -= sequence_.at(row, t_start - 1);
		}
		result.at(row, column) /= t_end - t_start + 1;
	}
}

void SegmentScorer::setSequence(const Math::Matrix<Float>& sequence) {
	sequence_.copyStructure(sequence);
	sequence_.copy(sequence);
	for (u32 col = 1; col < sequence_.nColumns(); col++) {
#pragma omp parallel for
		for (u32 row = 0; row < sequence_.nRows(); row++) {
			sequence_.at(row, col) += sequence_.at(row, col-1);
		}
	}
}

Float SegmentScorer::segmentScore(u32 t, u32 length, u32 c) {
	require(isInitialized_);
	u32 T = sequence_.nColumns();
	require_lt(t, T);
	u32 t_end = t;
	u32 t_start = t_end - length + 1;
	// are the scores already precomputed or do we have to compute them first?
	if ((t_start < t_start_) || (t_start > t_start_ + batchSize_ - 1) || (t_end != t_end_)) {
		// update for which t and t_start the scores are precomputed
		t_start_ = std::max((s32)t_end - (s32)batchSize_ + 1, 0);
		if (t_end - t_start + 1 > batchSize_)
			t_start_ = t_start;
		t_end_ = t_end;

		for (u32 column = 0; column < t_end_ - t_start_ + 1; column++) {
			generateVector(t_start_ + column, t_end_, column, networkInput_);
		}
		network_.outputLayer().initComputation(false);
		network_.forward(networkInput_);
		networkInput_.finishComputation(false);
		network_.outputLayer().finishComputation(true);
	}

	u32 column = (t_start - t_start_);
	require(!logarithmizeNetworkOutput_);
	Float segmentScore = std::log(network_.outputLayer().latestActivations(0).at(c, column)) - prior_.at(c);
	if (scaleSegmentByLength_)
		segmentScore *= length;
	return segmentScore;
}
