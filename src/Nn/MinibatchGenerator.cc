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
 * MinibatchGenerator.cc
 *
 *  Created on: May 17, 2016
 *      Author: richard
 */

#include <utility>
#include <algorithm>
#include "Types.hh"
#include "MatrixContainer.hh"
#include "MinibatchGenerator.hh"

using namespace Nn;

const Core::ParameterEnum MinibatchGenerator::paramSourceType_("source-type", "single, sequence", "single", "");

const Core::ParameterEnum MinibatchGenerator::paramTargetType_("target-type", "single, sequence", "single", "");

MinibatchGenerator::MinibatchGenerator(TrainingMode trainingMode) :
		sourceType_((FeatureType) Core::Configuration::config(paramSourceType_)),
		targetType_((FeatureType) Core::Configuration::config(paramTargetType_)),
		trainingMode_(trainingMode),
		featureReader_(0),
		sourceDimension_(0),
		targetDimension_(0),
		featureTransformation_(sourceType_),
		isInitialized_(false),
		generatedBatch_(false)
{}

MinibatchGenerator::~MinibatchGenerator() {
	if (featureReader_)
		delete featureReader_;
}

void MinibatchGenerator::initialize() {
	if (!isInitialized_) {
		if ((trainingMode_ == unsupervised) && (sourceType_ == single))
			featureReader_ = new Features::FeatureReader;
		else if ((trainingMode_ == unsupervised) && (sourceType_ == sequence))
			featureReader_ = new Features::SequenceFeatureReader;
		else if ((trainingMode_ == supervised) && (sourceType_ == single) && (targetType_ == single))
			featureReader_ = new Features::AlignedFeatureReader;
		else if ((trainingMode_ == supervised) && (sourceType_ == sequence) && (targetType_ == single))
			featureReader_ = new Features::AlignedSequenceFeatureReader;
		else if ((trainingMode_ == supervised) && (sourceType_ == sequence) && (targetType_ == sequence))
			featureReader_ = new Features::TemporallyAlignedSequenceFeatureReader;
		else {
			Core::Error::msg("MinibatchGenerator: Requested source/target pair is not implemented.") << Core::Error::abort;
		}

		featureReader_->initialize();
		sourceDimension_ = featureReader_->featureDimension();
		if ((trainingMode_ == supervised) && (sourceType_ == single) && (targetType_ == single))
			targetDimension_ = dynamic_cast< Features::AlignedFeatureReader* >(featureReader_)->targetDimension();
		else if ((trainingMode_ == supervised) && (sourceType_ == sequence) && (targetType_ == single))
			targetDimension_ = dynamic_cast< Features::AlignedSequenceFeatureReader* >(featureReader_)->targetDimension();
		else if ((trainingMode_ == supervised) && (sourceType_ == sequence) && (targetType_ == sequence))
			targetDimension_ = dynamic_cast< Features::TemporallyAlignedSequenceFeatureReader* >(featureReader_)->targetDimension();
		isInitialized_ = true;
	}
}

void MinibatchGenerator::read(std::vector< Math::Matrix<Float> >& source, std::vector< Math::Matrix<Float> >& target) {

	/* read source */
	if (sourceType_ == single) {
		source.push_back(Math::Matrix<Float>(sourceDimension_, 1));
		source.back().copy(dynamic_cast< Features::FeatureReader* >(featureReader_)->next().begin());
	}
	else {
		const Math::Matrix<Float>& tmp = dynamic_cast< Features::SequenceFeatureReader* >(featureReader_)->next();
		source.push_back(Math::Matrix<Float>(sourceDimension_, tmp.nColumns()));
		source.back().copy(tmp.begin());
	}

	/* read target */
	// (a) single targets
	if ((trainingMode_ == supervised) && (targetType_ == single)) {
		target.push_back(Math::Matrix<Float>(targetDimension_, 1));
		target.back().setToZero();
		if (sourceType_ == single)
			target.back().copy(dynamic_cast< Features::AlignedFeatureReader* >(featureReader_)->target().begin());
		else if (sourceType_ == sequence)
			target.back().copy(dynamic_cast< Features::AlignedSequenceFeatureReader* >(featureReader_)->target().begin());
	}
	// (b) sequence targets
	else if ((trainingMode_ == supervised) && (targetType_ == sequence)) {
		target.push_back(Math::Matrix<Float>(targetDimension_, source.back().nColumns())); // source and target sequence have same length
		target.back().copy(dynamic_cast< Features::TemporallyAlignedSequenceFeatureReader* >(featureReader_)->target());
	}
}

void MinibatchGenerator::generateSingleBatch(const std::vector< Math::Matrix<Float> >& source, const std::vector< Math::Matrix<Float> >& target) {
	// copy source features
	sourceBatch_.resize(sourceDimension_, source.size());
	for (u32 i = 0; i < source.size(); i++) {
		for (u32 d = 0; d < sourceDimension_; d++) {
			sourceBatch_.at(d, i) = source.at(i).at(d, 0);
		}
	}
	// copy target features
	targetBatch_.resize(targetDimension_, target.size());
	for (u32 i = 0; i < target.size(); i++) {
		for (u32 d = 0; d < targetDimension_; d++) {
			targetBatch_.at(d, i) = target.at(i).at(d, 0);
		}
	}
}

void MinibatchGenerator::generateSequenceBatch(const std::vector< Math::Matrix<Float> >& source, const std::vector< Math::Matrix<Float> >& target) {
	// sort source and target sequences from long to short
	std::vector< std::pair<u32, u32> > lengthsAndIndices;
	for (u32 i = 0; i < source.size(); i++)
		lengthsAndIndices.push_back(std::make_pair(source.at(i).nColumns(), i));
	std::stable_sort(lengthsAndIndices.begin(), lengthsAndIndices.end(), compare);
	order_.clear();
	for (u32 i = 0; i < lengthsAndIndices.size(); i++)
		order_.push_back(lengthsAndIndices.at(i).second);

	// how many sequences are active (started) at time frame t?
	u32 maxSequenceLength = source.at(order_.at(0)).nColumns();
	std::vector<u32> nStartedSequences(maxSequenceLength, 0);
	u32 i = 0;
	for (u32 t = 0; t < maxSequenceLength; t++) {
		if (t > 0)
			nStartedSequences.at(t) = nStartedSequences.at(t - 1);
		while ( (i < source.size()) && (maxSequenceLength - source.at(order_.at(i)).nColumns() == t) ) {
			nStartedSequences.at(t)++;
			i++;
		}
	}

	// add all time frames to the actual sequence source/target batch (note that the sequences are ordered!)
	sourceSequenceBatch_.reset();
	sourceSequenceBatch_.setMaximalMemory(maxSequenceLength);
	if ((trainingMode_ == supervised) && (targetType_ == sequence)) {
		targetSequenceBatch_.reset();
		targetSequenceBatch_.setMaximalMemory(maxSequenceLength);
	}
	for (u32 t = 0; t < maxSequenceLength; t++) {
		sourceSequenceBatch_.addTimeframe(sourceDimension_, nStartedSequences.at(t));
		if ((trainingMode_ == supervised) && (targetType_ == sequence))
			targetSequenceBatch_.addTimeframe(targetDimension_, nStartedSequences.at(t));
		for (u32 i = 0; i < nStartedSequences.at(t); i++) {
#pragma omp parallel for
			for (u32 d = 0; d < sourceDimension_; d++) {
				u32 offset = maxSequenceLength - source.at(order_.at(i)).nColumns();
				sourceSequenceBatch_.getLast().at(d, i) = source.at(order_.at(i)).at(d, t - offset);
			}
			if ((trainingMode_ == supervised) && (targetType_ == sequence)) {
#pragma omp parallel for
				for (u32 d = 0; d < targetDimension_; d++) {
					u32 offset = maxSequenceLength - target.at(order_.at(i)).nColumns();
					targetSequenceBatch_.getLast().at(d, i) = target.at(order_.at(i)).at(d, t - offset);
				}
			}
		}
	}

	// if targets are not sequences, add them to targetBatch_
	if ((trainingMode_ == supervised) && (targetType_ == single)) {
		targetBatch_.resize(targetDimension_, target.size());
		for (u32 i = 0; i < target.size(); i++) {
			for (u32 d = 0; d < targetDimension_; d++) {
				targetBatch_.at(d, i) = target.at(order_.at(i)).at(d, 0);
			}
		}
	}
}

void MinibatchGenerator::generateBatch(u32 batchSize) {
	require(isInitialized_);
	// make sure none of the containers is in computing state
	sourceBatch_.finishComputation(false);
	targetBatch_.finishComputation(false);
	sourceSequenceBatch_.finishComputation(false);
	targetSequenceBatch_.finishComputation(false);

	std::vector< Math::Matrix<Float> > source, target;
	if (sourceType_ == single) {
		Features::FeatureReader* fr = dynamic_cast< Features::FeatureReader* >(featureReader_);
		for (u32 i = 0; i < batchSize; i++) {
			if (!fr->hasFeatures())
				fr->newEpoch();
			read(source, target);
		}
	}
	else {
		Features::SequenceFeatureReader* fr = dynamic_cast< Features::SequenceFeatureReader* >(featureReader_);
		for (u32 i = 0; i < batchSize; i++) {
			if (!fr->hasSequences())
				fr->newEpoch();
			read(source, target);
		}
	}

	// throw error message if no features were available
	if (source.size() == 0)
		Core::Error::msg("MinibatchGenerator::generateBatch: feature reader has no observations left to fill a minibatch.") << Core::Error::abort;

	if (sourceType_ == single)
		generateSingleBatch(source, target);
	else
		generateSequenceBatch(source, target);

	// apply feature transformation depending on the type
	if ((sourceType_ == single) && (featureTransformation_.outputFormat() == sequence))
		featureTransformation_.transform(sourceBatch_, sourceSequenceBatch_);
	else if ((sourceType_ == sequence) && (featureTransformation_.outputFormat() == single))
		featureTransformation_.transform(sourceSequenceBatch_, sourceBatch_);

	generatedBatch_ = true;
}

u32 MinibatchGenerator::totalNumberOfFeatures() const {
	require(isInitialized_);
	return featureReader_->totalNumberOfFeatures();
}

u32 MinibatchGenerator::totalNumberOfObservations() const {
	require(isInitialized_);
	if (sourceType_ == single)
		return featureReader_->totalNumberOfFeatures();
	else
		return dynamic_cast< Features::SequenceFeatureReader* >(featureReader_)->totalNumberOfSequences();
}

Matrix& MinibatchGenerator::sourceBatch() {
	require(generatedBatch_);
	if (featureTransformation_.outputFormat() == sequence)
		Core::Error::msg("Method MinibatchGenerator::sourceBatch() is not available. Source is a sequence.") << Core::Error::abort;
	return sourceBatch_;
}

Matrix& MinibatchGenerator::targetBatch() {
	require(generatedBatch_);
	if (trainingMode_ == unsupervised)
		Core::Error::msg("Method MinibatchGenerator::targetBatch is only available for supervised training.") << Core::Error::abort;
	if (targetType_ == sequence)
		Core::Error::msg("Method MinibatchGenerator::targetBatch is not available if target type is sequence.") << Core::Error::abort;
	return targetBatch_;
}

MatrixContainer& MinibatchGenerator::sourceSequenceBatch() {
	require(generatedBatch_);
	if (featureTransformation_.outputFormat() == single)
		Core::Error::msg("Method MinibatchGenerator::sourceSequenceBatch() is not available. Source is not a sequence.") << Core::Error::abort;
	return sourceSequenceBatch_;
}

MatrixContainer& MinibatchGenerator::targetSequenceBatch() {
	require(generatedBatch_);
	if (trainingMode_ == unsupervised)
		Core::Error::msg("Method MinibatchGenerator::targetSequenceBatch is only available for supervised training.") << Core::Error::abort;
	if (targetType_ == single)
		Core::Error::msg("Method MinibatchGenerator::targetSequenceBatch is not available if target type is single.") << Core::Error::abort;
	return targetSequenceBatch_;
}
