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
 * LabeledFeatureReader.cc
 *
 *  Created on: Apr 11, 2014
 *      Author: richard
 */

#include "AlignedFeatureReader.hh"

using namespace Features;

/*
 * BaseAlignedFeatureReader
 */
// deprecated
const Core::ParameterString BaseAlignedFeatureReader::paramLabelCache_("label-cache", "", "features.labeled-feature-reader");

const Core::ParameterString BaseAlignedFeatureReader::paramTargetCache_("target-cache", "", "features.labeled-feature-reader");

BaseAlignedFeatureReader::BaseAlignedFeatureReader(const char* name) :
		targetCacheFile_(Core::Configuration::config(paramLabelCache_, name))
{
	if (!Core::Configuration::config(paramTargetCache_, name).empty()) {
		targetCacheFile_ = Core::Configuration::config(paramTargetCache_, name);
		if (!Core::Configuration::config(paramLabelCache_, name).empty()) {
			std::cout << "WARNING: " << name << ".label-cache and " << name << ".target-cache specified. Use parameter " << name
					<< ".target-cache: " << targetCacheFile_ << std::endl;
		}
	}
}

/*
 * AlignedFeatureReader
 */
AlignedFeatureReader::AlignedFeatureReader(const char* name) :
		Precursor(name),
		BaseAlignedFeatureReader(name),
		targetReader_(name, targetCacheFile_, bufferSize_, false)
{}

void AlignedFeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		targetReader_.initialize();
		target_.resize(targetReader_.featureDimension());
		if (totalNumberOfFeatures() != targetReader_.totalNumberOfFeatures()) {
			std::cerr << "Error: Number of observations in cache " << getCacheFilename() << " must match number of targets in "
					<< targetReader_.getCacheFilename() << ". Abort." << std::endl;
			exit(1);
		}
	}
}

void AlignedFeatureReader::shuffleIndices() {
	Precursor::shuffleIndices();
	targetReader_.reorderBufferedFeatures(reorderedIndices_);
}

void AlignedFeatureReader::newEpoch() {
	Precursor::newEpoch();
	targetReader_.newEpoch();
}

const Math::Vector<Float>& AlignedFeatureReader::next() {
	const Math::Vector<Float>& v = Precursor::next();
	target_.copy(targetReader_.next());
	return v;
}

u32 AlignedFeatureReader::targetDimension() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}

const Math::Vector<Float>& AlignedFeatureReader::target() const {
	require(isInitialized_);
	return target_;
}

/*
 * LabeledFeatureReader
 */
LabeledFeatureReader::LabeledFeatureReader(const char* name) :
		Precursor(name)
{}

void LabeledFeatureReader::initialize() {
	targetReader_.initialize();
	if (targetReader_.getFeatureType() != FeatureCache::labels) {
		Core::Error::msg("Target cache ") << targetReader_.getCacheFilename() << " is no label cache." << Core::Error::abort;
	}
	Precursor::initialize();
}

u32 LabeledFeatureReader::label() const {
	require(isInitialized_);
	return target_.argAbsMax();
}

u32 LabeledFeatureReader::nClasses() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}

/*
 * AlignedSequenceFeatureReader
 */
AlignedSequenceFeatureReader::AlignedSequenceFeatureReader(const char* name) :
		Precursor(name),
		BaseAlignedFeatureReader(name),
		targetReader_(name, targetCacheFile_, bufferSize_, false)
{}

void AlignedSequenceFeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		targetReader_.initialize();
		target_.resize(targetReader_.featureDimension());
		if (totalNumberOfSequences() != targetReader_.totalNumberOfFeatures()) {
			std::cerr << "Error: Number of sequences in cache " << getCacheFilename() << " must match number of targets in "
					<< targetReader_.getCacheFilename() << ". Abort." << std::endl;
			exit(1);
		}
	}
}

void AlignedSequenceFeatureReader::shuffleIndices() {
	Precursor::shuffleIndices();
	targetReader_.reorderBufferedFeatures(reorderedIndices_);
}

void AlignedSequenceFeatureReader::sortSequences() {
	Precursor::sortSequences();
	targetReader_.reorderBufferedFeatures(reorderedIndices_);
}

void AlignedSequenceFeatureReader::newEpoch() {
	Precursor::newEpoch();
	targetReader_.newEpoch();
}

const Math::Matrix<Float>& AlignedSequenceFeatureReader::next() {
	const Math::Matrix<Float>& v = Precursor::next();
	target_.copy(targetReader_.next());
	return v;
}

u32 AlignedSequenceFeatureReader::targetDimension() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}

const Math::Vector<Float>& AlignedSequenceFeatureReader::target() const {
	require(isInitialized_);
	return target_;
}

/*
 * LabeledSequenceFeatureReader
 */
LabeledSequenceFeatureReader::LabeledSequenceFeatureReader(const char* name) :
		Precursor(name)
{}

void LabeledSequenceFeatureReader::initialize() {
	targetReader_.initialize();
	if (targetReader_.getFeatureType() != FeatureCache::labels) {
		Core::Error::msg("Target cache ") << targetReader_.getCacheFilename() << " is no label cache." << Core::Error::abort;
	}
	Precursor::initialize();
}

u32 LabeledSequenceFeatureReader::label() const {
	require(isInitialized_);
	return target_.argAbsMax();
}

u32 LabeledSequenceFeatureReader::nClasses() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}

/*
 * TemporallyAlignedSequenceFeatureReader
 */
TemporallyAlignedSequenceFeatureReader::TemporallyAlignedSequenceFeatureReader(const char* name) :
		Precursor(name),
		BaseAlignedFeatureReader(name),
		targetReader_(name, targetCacheFile_, bufferSize_, false, false)
{}

void TemporallyAlignedSequenceFeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		targetReader_.initialize();
		if (totalNumberOfSequences() != targetReader_.totalNumberOfSequences()) {
			std::cerr << "Error: Number of sequences in cache " << getCacheFilename() << " must match number of target sequences in "
					<< targetReader_.getCacheFilename() << ". Abort." << std::endl;
			exit(1);
		}
		if (totalNumberOfFeatures() != targetReader_.totalNumberOfFeatures()) {
			std::cerr << "Error: Number of feature vectors in cache " << getCacheFilename() << " must match number of target vectors in "
					<< targetReader_.getCacheFilename() << ". Abort." << std::endl;
			exit(1);
		}
	}
}

void TemporallyAlignedSequenceFeatureReader::shuffleIndices() {
	Precursor::shuffleIndices();
	targetReader_.reorderBufferedFeatures(reorderedIndices_);
}

void TemporallyAlignedSequenceFeatureReader::sortSequences() {
	Precursor::sortSequences();
	targetReader_.reorderBufferedFeatures(reorderedIndices_);
}

void TemporallyAlignedSequenceFeatureReader::newEpoch() {
	Precursor::newEpoch();
	targetReader_.newEpoch();
}

const Math::Matrix<Float>& TemporallyAlignedSequenceFeatureReader::next() {
	const Math::Matrix<Float>& v = Precursor::next();
	const Math::Matrix<Float>& t = targetReader_.next();
	if (t.nColumns() != v.nColumns()) {
		std::cerr << "Error: Sequence lengths in cache " << getCacheFilename() << " do not match sequence lengths in target cache "
				<< targetReader_.getCacheFilename() << ". Abort." << std::endl;
		exit(1);
	}
	target_.resize(t.nRows(), t.nColumns());
	target_.copy(t);
	return v;
}

u32 TemporallyAlignedSequenceFeatureReader::targetDimension() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}

const Math::Matrix<Float>& TemporallyAlignedSequenceFeatureReader::target() const {
	require(isInitialized_);
	return target_;
}

/*
 * TemporallyLabeledSequenceFeatureReader
 */
TemporallyLabeledSequenceFeatureReader::TemporallyLabeledSequenceFeatureReader(const char* name) :
		Precursor(name)
{}

void TemporallyLabeledSequenceFeatureReader::initialize() {
	targetReader_.initialize();
	if (targetReader_.getFeatureType() != FeatureCache::sequencelabels) {
		Core::Error::msg("Target cache ") << targetReader_.getCacheFilename() << " is no label cache." << Core::Error::abort;
	}
	Precursor::initialize();
}

const Math::Matrix<Float>& TemporallyLabeledSequenceFeatureReader::next() {
	const Math::Matrix<Float>& v = Precursor::next();
	labelSequence_.clear();
	for (u32 i = 0; i < target_.nColumns(); i++) {
		labelSequence_.push_back(target_.argAbsMax(i));
	}
	return v;
}

const std::vector<u32>& TemporallyLabeledSequenceFeatureReader::labelSequence() const {
	require(isInitialized_);
	return labelSequence_;
}

u32 TemporallyLabeledSequenceFeatureReader::nClasses() const {
	require(isInitialized_);
	return targetReader_.featureDimension();
}
