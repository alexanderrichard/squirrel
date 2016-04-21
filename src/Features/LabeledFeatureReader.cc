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

#include "LabeledFeatureReader.hh"

using namespace Features;

/*
 * BaseLabeledFeatureReader
 */
const Core::ParameterString BaseLabeledFeatureReader::paramLabelCache_("label-cache", "", "features.labeled-feature-reader");

BaseLabeledFeatureReader::BaseLabeledFeatureReader(const char* name) :
		labelCacheFile_(Core::Configuration::config(paramLabelCache_, name))
{}

bool BaseLabeledFeatureReader::checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache) {
	bool fits = labelCache.cacheFormat() == FeatureCache::labelCache;
	fits = fits && (featureCache.cacheFormat() == FeatureCache::featureCache);
	return fits;
}

void BaseLabeledFeatureReader::initialize() {
	labelCache_.initialize(labelCacheFile_);
}

/*
 * LabeledFeatureReader
 */
LabeledFeatureReader::LabeledFeatureReader(const char* name) :
		Precursor(name),
		BaseLabeledFeatureReader(name),
		label_(0)
{}

void LabeledFeatureReader::initialize() {
	if (!Precursor::isInitialized_) {
		Precursor::initialize();
		BaseLabeledFeatureReader::initialize();
		bool ok = checkConsistency(cache_, labelCache_) && (labelCache_.featureType() == cache_.featureType());
		if (!ok) {
			std::cerr << "Features::LabeledFeatureReader: label-cache " << labelCacheFile_ <<
					" is not consistent with feature-cache " << cacheFile_ << ". Abort." << std::endl;
			exit(1);
		}
		labelBuffer_.resize(bufferSize_);
	}
}

void LabeledFeatureReader::bufferNext() {
	Precursor::bufferNext();
	// in case of sequence caches...
	if (labelCache_.featureType() == FeatureCache::sequence) {
		// ... possibly skip the sequence length ...
		if (labelCache_.nextIsSequenceLengthEntry()) {
			labelCache_.read_u32();
		}
		// ... skip the time stamp ...
		labelCache_.read_u32();
	}
	// read next label
	labelBuffer_.at(nBufferedFeatures_) = labelCache_.read_u32();
}

bool LabeledFeatureReader::checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache) {
	bool fits = BaseLabeledFeatureReader::checkConsistency(featureCache, labelCache);
	fits = fits && (labelCache.featureType() == featureCache.featureType());
	return fits;
}

void LabeledFeatureReader::newEpoch() {
	Precursor::newEpoch();
	if (labelCache_.cacheSize() > bufferSize_) {
		labelCache_.reset();
	}
}

u32 LabeledFeatureReader::label() {
	return labelBuffer_.at(currentFeatureIndex_);
}

/*
 * LabeledSequenceFeatureReader
 */
LabeledSequenceFeatureReader::LabeledSequenceFeatureReader(const char* name) :
		Precursor(name),
		BaseLabeledFeatureReader(name)
{}

void LabeledSequenceFeatureReader::initialize() {
	if (!Precursor::isInitialized_) {
		Precursor::initialize();
		BaseLabeledFeatureReader::initialize();
		if (!checkConsistency(cache_, labelCache_)) {
			std::cerr << "Features::LabeledSequenceFeatureReader: label-cache " << labelCacheFile_ <<
					" is not consistent with feature-cache " << cacheFile_ << ". Abort." << std::endl;
			exit(1);
		}
		labelBuffer_.resize(bufferSize_);
	}
}

void LabeledSequenceFeatureReader::bufferNext() {
	Precursor::bufferNext();
	// read label for next sequence
	labelBuffer_.at(nBufferedFeatures_) = labelCache_.read_u32();
}

bool LabeledSequenceFeatureReader::checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache) {
	bool fits = BaseLabeledFeatureReader::checkConsistency(featureCache, labelCache);
	fits = fits && (labelCache.featureType() == FeatureCache::single);
	fits = fits && (featureCache.featureType() == FeatureCache::sequence);
	fits = fits && (labelCache.cacheSize() == featureCache.nSequences());
	return fits;
}

void LabeledSequenceFeatureReader::newEpoch() {
	Precursor::newEpoch();
	if (labelCache_.cacheSize() > bufferSize_) {
		labelCache_.reset();
	}
}

u32 LabeledSequenceFeatureReader::label() {
	return labelBuffer_.at(currentFeatureIndex_);
}

/*
 * TemporallyLabeledSequenceFeatureReader
 */
TemporallyLabeledSequenceFeatureReader::TemporallyLabeledSequenceFeatureReader(const char* name) :
		Precursor(name),
		BaseLabeledFeatureReader(name)
{}

void TemporallyLabeledSequenceFeatureReader::initialize() {
	if (!Precursor::isInitialized_) {
		Precursor::initialize();
		BaseLabeledFeatureReader::initialize();
		bool ok = checkConsistency(cache_, labelCache_) && (labelCache_.featureType() == cache_.featureType());
		if (!ok) {
			std::cerr << "Features::TemporallyLabeledSequenceFeatureReader: label-cache " << labelCacheFile_ <<
					" is not consistent with feature-cache " << cacheFile_ << ". Abort." << std::endl;
			exit(1);
		}
		labelBuffer_.resize(bufferSize_);
	}
}

void TemporallyLabeledSequenceFeatureReader::bufferNext() {
	Precursor::bufferNext();
	// check the sequence length
	if (currentSequenceLength_ != labelCache_.read_u32()) {
		std::cerr << "Features::LabeledSequenceFeatureReader: sequence lengths of label-cache " << labelCacheFile_ <<
				" and feature-cache " << cacheFile_ << " do not match. Abort." << std::endl;
		exit(1);
	}
	// read time stamps and labels of the sequence
	labelBuffer_.at(nBufferedFeatures_).resize(currentSequenceLength_);
	// read next sequence
	for (u32 t = 0; t < currentSequenceLength_; t++) {
		if (timestamps_.at(nBufferedFeatures_).at(t) != labelCache_.read_u32()) {
			std::cerr << "Features::LabeledSequenceFeatureReader: timestamps of label-cache " << labelCacheFile_ <<
					" and feature-cache " << cacheFile_ << " do not match. Abort." << std::endl;
			exit(1);
		}
		labelBuffer_.at(nBufferedFeatures_).at(t) = labelCache_.read_u32();
	}
}

bool TemporallyLabeledSequenceFeatureReader::checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache) {
	bool fits = BaseLabeledFeatureReader::checkConsistency(featureCache, labelCache);
	fits = fits && (labelCache.featureType() == FeatureCache::sequence);
	fits = fits && (featureCache.featureType() == FeatureCache::sequence);
	fits = fits && (labelCache.nSequences() == featureCache.nSequences());
	fits = fits && (labelCache.cacheSize() == featureCache.cacheSize());
	return fits;
}

void TemporallyLabeledSequenceFeatureReader::newEpoch() {
	Precursor::newEpoch();
	if (labelCache_.nSequences() > bufferSize_) {
		labelCache_.reset();
	}
}

const std::vector<u32>& TemporallyLabeledSequenceFeatureReader::labelSequence() {
	return labelBuffer_.at(currentFeatureIndex_);
}
