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
 * FeatureReader.cc
 *
 *  Created on: Apr 8, 2014
 *      Author: richard
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

BaseFeatureReader::BaseFeatureReader(const char* name, const std::string& cacheFile, u32 bufferSize,
		bool shuffleBuffer, const std::vector<std::string>& preprocessors) :
		name_(name),
		cacheFile_(cacheFile),
		nBufferedFeatures_(0),
		nProcessedFeatures_(0),
		nRemainingFeaturesInCache_(0),
		bufferSize_(bufferSize),
		shuffleBuffer_(shuffleBuffer),
		reorderBuffer_(shuffleBuffer),
		reorderedIndices_(0),
		nextFeatureIndex_(0),
		currentFeatureIndex_(0),
		preprocessorNames_(preprocessors),
		isInitialized_(false)
{}

void BaseFeatureReader::initialize() {
	if (cacheFile_.empty()) {
		std::cerr << "FeatureReader " << name_ << ": no cache file specified. Abort." << std::endl;
		exit(1);
	}
	// initialize cache file
	cache_.initialize(cacheFile_);
	// create and initialize preprocessors
	if ( ((cache_.featureType() == FeatureCache::labels) || (cache_.featureType() == FeatureCache::sequencelabels) )
			&& (preprocessorNames_.size() != 0)) {
		Core::Error::msg("BaseFeatureReader::initialize: Label cache ") << cacheFile_ << " must not have any preprocessors." << Core::Error::abort;
	}
	for (u32 i = 0; i < preprocessorNames_.size(); i++) {
		Preprocessor* p = Preprocessor::createPreprocessor(preprocessorNames_.at(i).c_str());
		if (preprocessors_.size() == 0)
			p->initialize(cache_.featureDim());
		else
			p->initialize(preprocessors_.back()->outputDimension());
		preprocessors_.push_back(p);
	}
}

void BaseFeatureReader::reorderBufferedFeatures(const std::vector<u32>& reorderedIndices) {
	reorderedIndices_ = reorderedIndices;
	reorderBuffer_ = true;
}

const std::vector<u32>& BaseFeatureReader::getReordering() const {
	return reorderedIndices_;
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
	// read next feature vector
	switch (cache_.featureType()) {
	case FeatureCache::vectors:
	case FeatureCache::images:
	case FeatureCache::labels:
		{
		const Math::Matrix<Float>& next = cache_.next();
		f.resize(next.nRows(), next.nColumns());
		f.copy(next);
		}
		break;
	default:
		Core::Error::msg("BaseFeatureReader::readFeatureVector: feature cache must not contain sequences.") << Core::Error::abort;
		break;
	}
}

void BaseFeatureReader::readFeatureSequence(Math::Matrix<Float>& f) {
	// read next feature vector
	switch (cache_.featureType()) {
	case FeatureCache::sequences:
	case FeatureCache::videos:
	case FeatureCache::sequencelabels:
		{
		const Math::Matrix<Float>& next = cache_.next();
		f.resize(next.nRows(), next.nColumns());
		f.copy(next);
		}
		break;
	default:
		Core::Error::msg("BaseFeatureReader::readFeatureSequence: feature cache does not contain sequences.") << Core::Error::abort;
		break;
	}
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
	std::random_shuffle(reorderedIndices_.begin(), reorderedIndices_.end(), Math::Random::randomIntBelow);
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

const std::string& BaseFeatureReader::getCacheFilename() const {
	return cacheFile_;
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
 * FeatureReader
 */

FeatureReader::FeatureReader(const char* name) :
		Precursor(name),
		buffer_(bufferSize_),
		needsContext_(false),
		prebufferPointer_(0)
{}

FeatureReader::FeatureReader(const char* name, const std::string& cacheFile, u32 bufferSize, bool shuffleBuffer,
		const std::vector<std::string>& preprocessors) :
		Precursor(name, cacheFile, bufferSize, shuffleBuffer, preprocessors),
		buffer_(bufferSize_),
		needsContext_(false),
		prebufferPointer_(0)
{}

void FeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		if (shuffleBuffer_)
			Math::Random::initializeSRand();
		nRemainingFeaturesInCache_ = cache_.cacheSize();
		if (bufferSize_ == 0) {
			Core::Log::openTag("feature-reader");
			Core::Log::os("buffer size is 0. Set buffer size to ") << cache_.cacheSize() << (" (#features in cache).");
			Core::Log::closeTag();
			bufferSize_ = cache_.cacheSize();
			buffer_.resize(bufferSize_);
		}
		labelBuffer_.resize(cache_.featureDim());
		// if the cache has sequence format, read whole sequences at once
		needsContext_ = (cache_.featureType() == FeatureCache::sequences) || (cache_.featureType() == FeatureCache::videos) || (cache_.featureType() == FeatureCache::sequencelabels);
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
		if ((cache_.featureType() == FeatureCache::labels) || (cache_.featureType() == FeatureCache::sequencelabels)) {
			buffer_.at(nBufferedFeatures_).resize(1);
			buffer_.at(nBufferedFeatures_).at(0) = out.argAbsMax(0);
		}
		else {
			buffer_.at(nBufferedFeatures_).swap(out);
		}
	}
	else {
		/* ensure the feature cache is a sequence cache */
		if (! ((cache_.featureType() == FeatureCache::sequences) || (cache_.featureType() == FeatureCache::videos) || (cache_.featureType() == FeatureCache::sequencelabels)) )
			Core::Error::msg("FeatureReader: Preprocessors need sequence context but cache-file is not in sequence format.") << Core::Error::abort;
		/* if context required: read a sequence */
		if (prebufferPointer_ == 0) {
			readFeatureSequence(in);
			applyPreprocessors(in, prebufferedSequence_);
		}
		// store result in buffer
		if ((cache_.featureType() == FeatureCache::labels) || (cache_.featureType() == FeatureCache::sequencelabels)) {
			buffer_.at(nBufferedFeatures_).resize(1);
			buffer_.at(nBufferedFeatures_).at(0) = prebufferedSequence_.argAbsMax(prebufferPointer_);
		}
		else {
			prebufferedSequence_.getColumn(prebufferPointer_, buffer_.at(nBufferedFeatures_));
		}
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
	if ((cache_.featureType() == FeatureCache::labels) || (cache_.featureType() == FeatureCache::sequencelabels)) {
		labelBuffer_.setToZero();
		labelBuffer_.at((u32)buffer_.at(index).at(0)) = 1.0;
		return labelBuffer_;
	}
	else {
		return buffer_.at(index);
	}
}

/*
 * SequenceFeatureReader
 */
const Core::ParameterBool SequenceFeatureReader::paramSortSequences_("sort-sequences", false, "features.feature-reader");

SequenceFeatureReader::SequenceFeatureReader(const char* name) :
		Precursor(name),
		buffer_(bufferSize_),
		currentSequenceLength_(0),
		sortSequences_(Core::Configuration::config(paramSortSequences_, name_))
{
	if (shuffleBuffer_ && sortSequences_) {
		std::cerr << "SequenceFeatureReader: shuffle-buffer and sort-sequences cannot be selected at the same time" << std::endl;
		exit(1);
	}
	reorderBuffer_ = (reorderBuffer_ || sortSequences_);
}

SequenceFeatureReader::SequenceFeatureReader(const char* name, const std::string& cacheFile, u32 bufferSize, bool shuffleBuffer,
		bool sortSequences, const std::vector<std::string>& preprocessors) :
		Precursor(name, cacheFile, bufferSize, shuffleBuffer, preprocessors),
		buffer_(bufferSize_),
		currentSequenceLength_(0),
		sortSequences_(sortSequences)
{}

void SequenceFeatureReader::initialize() {
	if (!isInitialized_) {
		Precursor::initialize();
		if (shuffleBuffer_)
			Math::Random::initializeSRand();
		if ((!cache_.featureType() == FeatureCache::sequences) || (!cache_.featureType() == FeatureCache::videos) || (!cache_.featureType() == FeatureCache::sequencelabels)) {
			std::cerr << "Error: Cache " << cacheFile_ << " is not in sequence format. Abort." << std::endl;
			exit(1);
		}
		nRemainingFeaturesInCache_ = cache_.nSequences();
		if (bufferSize_ == 0) {
			Core::Log::openTag("feature-reader");
			Core::Log::os("buffer size is 0. Set buffer size to ") << cache_.nSequences() << (" (#sequences in cache).");
			Core::Log::closeTag();
			bufferSize_ = cache_.nSequences();
			buffer_.resize(bufferSize_);
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
	readFeatureSequence(in);
	currentSequenceLength_ = in.nColumns();
	if (cache_.featureType() == FeatureCache::sequencelabels) {
		Math::Vector<u32> tmp(in.nColumns());
		in.argMax(tmp);
		buffer_.at(nBufferedFeatures_).resize(1, in.nColumns());
		buffer_.at(nBufferedFeatures_).copy(tmp.begin());
	}
	else {
		applyPreprocessors(in, buffer_.at(nBufferedFeatures_));
	}
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

const Math::Matrix<Float>& SequenceFeatureReader::next() {
	// there must be at least one more feature sequence to process
	require(nProcessedFeatures_ < cache_.nSequences());
	u32 index = nextFeature();
	currentFeatureIndex_ = index;
	// if feature cache
	if (cache_.featureType() == FeatureCache::sequencelabels) {
		labelBuffer_.resize(cache_.featureDim(), buffer_.at(index).nColumns());
		labelBuffer_.setToZero();
		for (u32 col = 0; col < labelBuffer_.nColumns(); col++)
			labelBuffer_.at(buffer_.at(index).at(0, col), col) = 1.0;
		return labelBuffer_;
	}
	else {
		return buffer_.at(index);
	}
}

/*
 * LabelReader
 */
LabelReader::LabelReader(const char* name) :
		Precursor(name)
{}

LabelReader::LabelReader(const char* name, const std::string& cacheFile, u32 bufferSize, bool shuffleBuffer) :
		Precursor(name, cacheFile, bufferSize, shuffleBuffer)
{}

void LabelReader::initialize() {
	Precursor::initialize();
	// validate that the cache is a label cache
	if (cache_.featureType() != FeatureCache::labels) {
		Core::Error::msg("Cache ") << cacheFile_ << " is not a label cache." << Core::Error::abort;
	}
	if (preprocessors_.size() != 0) {
		Core::Error::msg("Error: LabelReader must not have any preprocessors.") << Core::Error::abort;
	}
}

u32 LabelReader::nextLabel() {
	const Math::Vector<Float>& v = Precursor::next();
	return v.argAbsMax();
}

/*
 * SequenceLabelReader
 */
SequenceLabelReader::SequenceLabelReader(const char* name) :
		Precursor(name)
{}

SequenceLabelReader::SequenceLabelReader(const char* name, const std::string& cacheFile, u32 bufferSize, bool shuffleBuffer, bool sortSequences) :
		Precursor(name, cacheFile, bufferSize, shuffleBuffer, sortSequences)
{}

void SequenceLabelReader::initialize() {
	Precursor::initialize();
	// validate that the cache is a label cache
	if (cache_.featureType() != FeatureCache::sequencelabels) {
		Core::Error::msg("Cache ") << cacheFile_ << " is not a sequence label cache." << Core::Error::abort;
	}
	if (preprocessors_.size() != 0) {
		Core::Error::msg("Error: SequenceLabelReader must not have any preprocessors.") << Core::Error::abort;
	}
}

const std::vector<u32>& SequenceLabelReader::nextLabelSequence() {
	const Math::Matrix<Float>& m = Precursor::next();
	labels_.resize(m.nColumns());
	for (u32 i = 0; i < labels_.size(); i++) {
		labels_.at(i) = m.argAbsMax(i);
	}
	return labels_;
}
