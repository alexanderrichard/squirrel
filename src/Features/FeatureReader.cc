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

#include "FeatureReader.hh"
#include "Math/Random.hh"
#include "Math/Matrix.hh"
#include <algorithm>
#include <typeinfo>

using namespace Features;

/*
 * BaseFeatureReader
 */
const Core::ParameterString BaseFeatureReader::paramCacheFile_("feature-cache", "", "features.feature-reader");

const Core::ParameterInt BaseFeatureReader::paramBufferSize_("buffer-size", 0, "features.feature-reader");

const Core::ParameterBool BaseFeatureReader::paramShuffleBuffer_("shuffle-buffer", false, "features.feature-reader");

const Core::ParameterStringList BaseFeatureReader::paramPreprocessingSteps_("preprocessors", "", "features.feature-reader");

BaseFeatureReader::BaseFeatureReader(const char* name) :
		name_(name),
		cacheFile_(Core::Configuration::config(paramCacheFile_, name_)),
		nBufferedFeatures_(0),
		nProcessedFeatures_(0),
		nRemainingFeaturesInCache_(0),
		bufferSize_(Core::Configuration::config(paramBufferSize_, name_)),
		shuffleBuffer_(Core::Configuration::config(paramShuffleBuffer_, name_)),
		reorderBuffer_(shuffleBuffer_),
		reorderedIndices_(0),
		nextFeatureIndex_(0),
		currentFeatureIndex_(0),
		preprocessorNames_(Core::Configuration::config(paramPreprocessingSteps_, name_)),
		isInitialized_(false)
{}

void BaseFeatureReader::initialize() {
	if (cacheFile_.empty()) {
		std::cerr << "FeatureReader " << name_ << ": no cache file specified. Abort." << std::endl;
		exit(1);
	}
	// initialize cache file
	cache_.initialize(cacheFile_);
	// validate that the cache is a feature cache
	validateCacheFormat();
	// create and initialize preprocessors
	for (u32 i = 0; i < preprocessorNames_.size(); i++) {
		Preprocessor* p = Preprocessor::createPreprocessor(preprocessorNames_.at(i).c_str());
		if (preprocessors_.size() == 0)
			p->initialize(cache_.featureDim());
		else
			p->initialize(preprocessors_.back()->outputDimension());
		preprocessors_.push_back(p);
	}
}

void BaseFeatureReader::resetBuffer() {
	nextFeatureIndex_ = 0;
}

void BaseFeatureReader::fillBuffer() {
	resetBuffer();
	nBufferedFeatures_ = 0;
	// local copy needed because bufferNext modifies nRemainingFeaturesInCache_
	u32 nRemainingFeaturesInCache = nRemainingFeaturesInCache_;
	for (u32 i = 0; i < std::min(bufferSize_, nRemainingFeaturesInCache); i++) {
		bufferNext();
		// update variables
		nBufferedFeatures_++;
		nRemainingFeaturesInCache_--;
	}
}

void BaseFeatureReader::readFeatureVector(Math::Matrix<Float>& f) {
	// in case of sequence caches...
	if (cache_.featureType() == FeatureCache::sequence) {
		// ... skip the sequence length ...
		if (cache_.nextIsSequenceLengthEntry()) {
			cache_.read_u32();
		}
		// ... skip the time stamp ...
		cache_.read_u32();
	}
	// read next feature vector
	f.resize(cache_.featureDim(), 1);
	for (u32 i = 0; i < cache_.featureDim(); i++) {
		f.at(i,0) = cache_.read_Float();
	}
}

void BaseFeatureReader::readFeatureSequence(Math::Matrix<Float>& f, std::vector<u32>& timestamps) {
	// read the sequence length
	u32 sequenceLength = cache_.read_u32();
	// read time stamps and feature vectors of the sequence
	timestamps.resize(sequenceLength);
	// read next sequence (each column contains a feature vector)
	f.resize(cache_.featureDim(), sequenceLength);
	for (u32 t = 0; t < sequenceLength; t++) {
		timestamps.at(t) = cache_.read_u32();
		for (u32 d = 0; d < cache_.featureDim(); d++) {
			f.at(d, t) = cache_.read_Float();
		}
	}
}

void BaseFeatureReader::readFeatureSequence(Math::Matrix<Float>& f) {
	std::vector<u32> dummy;
	readFeatureSequence(f, dummy);
}

void BaseFeatureReader::applyPreprocessors(Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	for (u32 p = 0; p < preprocessors_.size(); p++) {
		preprocessors_.at(p)->work(in, out);
		in.swap(out);
	}
	out.swap(in);
}

void BaseFeatureReader::shuffleIndices() {
	reorderedIndices_.resize(nBufferedFeatures_);
	for (u32 i = 0; i < nBufferedFeatures_; i++) {
		reorderedIndices_[i] = i;
	}
	std::random_shuffle(reorderedIndices_.begin(), reorderedIndices_.end());
}

void BaseFeatureReader::validateCacheFormat() {
	if (cache_.cacheFormat() != FeatureCache::featureCache) {
		std::cerr << "BaseFeatureReader::validateCacheFormat: cache is not of type feature-cache. Abort." << std::endl;
		exit(1);
	}
}

u32 BaseFeatureReader::nextFeature() {
	require(isInitialized_);
	// there must be at least one more feature vector to process
	require(nProcessedFeatures_ < cache_.cacheSize());
	// check if buffer needs to be refilled
	if (allBufferedFeaturesRead()) {
		fillBuffer();
		// new random order of elements in buffer
		if (shuffleBuffer_) {
			shuffleIndices();
		}
	}
	require_lt(nextFeatureIndex_, nBufferedFeatures_);
	u32 index = nextFeatureIndex_;
	if (reorderBuffer_) {
		index = reorderedIndices_.at(nextFeatureIndex_);
	}
	nProcessedFeatures_++;
	nextFeatureIndex_++;
	return index;
}

bool BaseFeatureReader::allBufferedFeaturesRead() {
	return (nextFeatureIndex_ == nBufferedFeatures_);
}

u32 BaseFeatureReader::maxBufferSize() const {
	return bufferSize_;
}

FeatureCache::FeatureType BaseFeatureReader::getFeatureType() const {
	return cache_.featureType();
}

FeatureCache::CacheFormat BaseFeatureReader::getCacheFormat() const {
	return cache_.cacheFormat();
}

bool BaseFeatureReader::shuffleBuffer() const {
	return shuffleBuffer_;
}

u32 BaseFeatureReader::totalNumberOfFeatures() const {
	require(isInitialized_);
	return cache_.cacheSize();
}

u32 BaseFeatureReader::featureDimension() const {
	require(isInitialized_);
	if (preprocessors_.size() > 0)
		return preprocessors_.back()->outputDimension();
	else
		return cache_.featureDim();
}

void BaseFeatureReader::newEpoch() {
	require(isInitialized_);
	resetBuffer();
	// if cache does not fit completely into buffer...
	if  (   ((!isSequenceReader())   && (cache_.cacheSize() > bufferSize_))   ||
			(  isSequenceReader()    && (cache_.nSequences() > bufferSize_))   ) {
		cache_.reset();
		nBufferedFeatures_ = 0;
	}
	// generate a new random order
	if (shuffleBuffer_) {
		shuffleIndices();
	}
	nProcessedFeatures_ = 0;
	nRemainingFeaturesInCache_ = cache_.cacheSize();
}

/*
 * HeaderReader
 */
HeaderReader::HeaderReader(const char* name) :
		Precursor(name)
{}

void HeaderReader::initialize() {
	if (!isInitialized_) {
		if (cacheFile_.empty()) {
			std::cerr << "FeatureReader " << name_ << ": no cache file specified. Abort." << std::endl;
			exit(1);
		}
		cache_.setLogCacheInformation(false);
		cache_.initialize(cacheFile_);
		isInitialized_ = true;
	}
}

/*
 * FeatureReader
 */

FeatureReader::FeatureReader(const char* name) :
		Precursor(name),
		buffer_(bufferSize_),
		needsContext_(false),
		prebufferPointer_(0)
{}

void FeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		if (shuffleBuffer_)
			Math::RandomNumberGenerator::initializeSRand();
		nRemainingFeaturesInCache_ = cache_.cacheSize();
		if (bufferSize_ == 0) {
			Core::Log::openTag("feature-reader");
			Core::Log::os("buffer size is 0. Set buffer size to ") << cache_.cacheSize() << (" (#features in cache).");
			Core::Log::closeTag();
			bufferSize_ = cache_.cacheSize();
			buffer_.resize(bufferSize_);
		}
		// check if there is a preprocessor that needs sequence context
		for (u32 p = 0; p < preprocessors_.size(); p++) {
			needsContext_ = needsContext_ || preprocessors_.at(p)->needsContext();
		}
		isInitialized_ = true;
	}
}

void FeatureReader::bufferNext() {
	require(nRemainingFeaturesInCache_ > 0);
	require(nBufferedFeatures_ < bufferSize_);
	Math::Matrix<Float> in;

	if (!needsContext_) {
		/* if no context required: read a single feature vector */
		Math::Matrix<Float> out;
		readFeatureVector(in);
		applyPreprocessors(in, out);
		// store result in buffer
		buffer_.at(nBufferedFeatures_).swap(out);
	}
	else {
		/* ensure the feature cache is a sequence cache */
		if (!cache_.featureType() == FeatureCache::sequence) {
			std::cerr << "FeatureReader: Preprocessors need sequence context but cache-file is not in sequence format. Abort." << std::endl;
			exit(1);
		}
		/* if context required: read a sequence */
		if (prebufferPointer_ == 0) {
			readFeatureSequence(in);
			applyPreprocessors(in, prebufferedSequence_);
		}
		prebufferedSequence_.getColumn(prebufferPointer_, buffer_.at(nBufferedFeatures_));
		prebufferPointer_++;
		if (prebufferPointer_ >= prebufferedSequence_.nColumns())
			prebufferPointer_ = 0;
	}
}

bool FeatureReader::hasFeatures() const {
	require(isInitialized_);
	return (nProcessedFeatures_ < cache_.cacheSize());
}

const Math::Vector<Float>& FeatureReader::next() {
	u32 index = nextFeature();
	currentFeatureIndex_ = index;
	return buffer_.at(index);
}

/*
 * SequenceFeatureReader
 */
const Core::ParameterBool SequenceFeatureReader::paramSortSequences_("sort-sequences", false, "features.feature-reader");

SequenceFeatureReader::SequenceFeatureReader(const char* name) :
		Precursor(name),
		buffer_(bufferSize_),
		timestamps_(bufferSize_),
		currentSequenceLength_(0),
		sortSequences_(Core::Configuration::config(paramSortSequences_, name_))
{
	if (shuffleBuffer_ && sortSequences_) {
		std::cerr << "SequenceFeatureReader: shuffle-buffer and sort-sequences cannot be selected at the same time" << std::endl;
		exit(1);
	}
	reorderBuffer_ = (reorderBuffer_ || sortSequences_);
}

void SequenceFeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		if (shuffleBuffer_)
			Math::RandomNumberGenerator::initializeSRand();
		if (cache_.featureType() != FeatureCache::sequence) {
			std::cerr << "SequenceFeatureReader: Cache is not in sequence format. Abort." << std::endl;
			exit(1);
		}
		nRemainingFeaturesInCache_ = cache_.nSequences();
		if (bufferSize_ == 0) {
			Core::Log::openTag("feature-reader");
			Core::Log::os("buffer size is 0. Set buffer size to ") << cache_.nSequences() << (" (#sequences in cache).");
			Core::Log::closeTag();
			bufferSize_ = cache_.nSequences();
			buffer_.resize(bufferSize_);
			timestamps_.resize(bufferSize_);
		}
		isInitialized_ = true;
	}
}

void SequenceFeatureReader::sortSequences() {
	reorderedIndices_.resize(nBufferedFeatures_);
	for (u32 i = 0; i < nBufferedFeatures_; i++) {
		reorderedIndices_.at(i) = i;
	}
	std::sort(reorderedIndices_.begin(), reorderedIndices_.end(), SequenceLengthComparator(this));
}

void SequenceFeatureReader::fillBuffer() {
	Precursor::fillBuffer();
	if (sortSequences_) {
		sortSequences();
	}
}

void SequenceFeatureReader::bufferNext() {
	require(nRemainingFeaturesInCache_ > 0);
	require(nBufferedFeatures_ < bufferSize_);

	// read a feature sequence
	Math::Matrix<Float> in;
	readFeatureSequence(in, timestamps_.at(nBufferedFeatures_));
	currentSequenceLength_ = timestamps_.at(nBufferedFeatures_).size();
	applyPreprocessors(in, buffer_.at(nBufferedFeatures_));
}

u32 SequenceFeatureReader::totalNumberOfSequences() const {
	return cache_.nSequences();
}

bool SequenceFeatureReader::hasSequences() const {
	require(isInitialized_);
	return (nProcessedFeatures_ < cache_.nSequences());
}

bool SequenceFeatureReader::areSequencesSorted() const {
	return sortSequences_;
}

void SequenceFeatureReader::newEpoch() {
	Precursor::newEpoch();
	nRemainingFeaturesInCache_ = cache_.nSequences();
}

std::vector<u32>& SequenceFeatureReader::currentTimestamps() {
	return timestamps_.at(currentFeatureIndex_);
}

const Math::Matrix<Float>& SequenceFeatureReader::next() {
	// there must be at least one more feature sequence to process
	require(nProcessedFeatures_ < cache_.nSequences());
	u32 index = nextFeature();
	currentFeatureIndex_ = index;
	return buffer_.at(index);
}

/*
 * LabelReader
 */
void LabelReader::validateCacheFormat() {
	if (cache_.cacheFormat() != FeatureCache::labelCache) {
		std::cerr << "LabelReader::validateCacheFormat: cache is not of type label-cache. Abort." << std::endl;
		exit(1);
	}
}

void LabelReader::initialize() {
	Precursor::initialize();
	if (preprocessors_.size() != 0) {
		std::cerr << "Error: LabelReader must not have any preprocessors. Abort." << std::endl;
		exit(1);
	}
}

u32 LabelReader::next() {
	const Math::Vector<Float>& v = Precursor::next();
	require_eq(v.nRows(), 1);
	Float f = v.at(0);
	u32 label = reinterpret_cast<u32 &>(f);
	return label;
}

/*
 * SequenceLabelReader
 */
void SequenceLabelReader::validateCacheFormat() {
	if (cache_.cacheFormat() != FeatureCache::labelCache) {
		std::cerr << "SequenceLabelReader::validateCacheFormat: cache is not of type label-cache. Abort." << std::endl;
		exit(1);
	}
}

void SequenceLabelReader::initialize() {
	Precursor::initialize();
	if (preprocessors_.size() != 0) {
		std::cerr << "Error: LabelReader must not have any preprocessors. Abort." << std::endl;
		exit(1);
	}
}

std::vector<u32>& SequenceLabelReader::next() {
	const Math::Matrix<Float>& m = Precursor::next();
	require_eq(m.nRows(), 1);
	labels_.resize(m.nColumns());
	for (u32 i = 0; i < labels_.size(); i++) {
		Float f = m.at(0, i);
		labels_.at(i) = reinterpret_cast<u32 &>(f);
	}
	return labels_;
}
