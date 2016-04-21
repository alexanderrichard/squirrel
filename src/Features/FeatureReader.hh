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

#ifndef FEATURES_FEATUREREADER_HH_
#define FEATURES_FEATUREREADER_HH_

#include "Core/CommonHeaders.hh"
#include "FeatureCache.hh"
#include "Preprocessor.hh"
#include <string.h>

namespace Features {

/**
 * BaseFeatureReader
 */
class BaseFeatureReader
{
private:
	static const Core::ParameterString paramCacheFile_;
	static const Core::ParameterInt paramBufferSize_;
	static const Core::ParameterBool paramShuffleBuffer_;
	static const Core::ParameterStringList paramPreprocessingSteps_;
protected:
	const char* name_;				// name of the feature reader (important, if multiple feature readers exist)

	FeatureCache cache_;
	std::string cacheFile_;
	u32 nBufferedFeatures_;
	u32 nProcessedFeatures_;		// number of features from the cache that have been processed
	u32 nRemainingFeaturesInCache_;	// number of features in the cache that have not yet been read
	u32 bufferSize_;
	bool shuffleBuffer_;
	bool reorderBuffer_;
	std::vector<u32> reorderedIndices_;		// indeces from 0 to nBufferedFeatures_ - 1 in another order
	u32 nextFeatureIndex_;					// pointer to the next feature to be returned by next()
	u32 currentFeatureIndex_;				// points to the index in the buffer that is recently read

	std::vector<std::string> preprocessorNames_;	// names of the preprocessors
	std::vector<Preprocessor*> preprocessors_;		// sequence of preprocessors applied to the features

	bool isInitialized_;

	void resetBuffer();
	virtual void fillBuffer();
	void shuffleIndices();
	bool allBufferedFeaturesRead();
	void readFeatureVector(Math::Matrix<Float>& f);
	void readFeatureSequence(Math::Matrix<Float>& f, std::vector<u32>& timestamps);
	void readFeatureSequence(Math::Matrix<Float>&f);
	void applyPreprocessors(Math::Matrix<Float>& in, Math::Matrix<Float>& out);
	virtual void bufferNext() = 0;
	virtual bool isSequenceReader() = 0;
	virtual void validateCacheFormat();		// validate the cache format (requires feature cache by default)
	u32 nextFeature();				// get index of next feature (and fill buffer, shuffle indices)
public:
	BaseFeatureReader(const char* name);
	virtual ~BaseFeatureReader() {}

	virtual void initialize();

	/*
	 * @return the size of the buffer as set by the parameter
	 */
	u32 maxBufferSize() const;

	/*
	 * @return the feature type (single or sequence)
	 */
	FeatureCache::FeatureType getFeatureType() const;

	/*
	 * @return the cache format (feature-cache or label-cache)
	 */
	FeatureCache::CacheFormat getCacheFormat() const;

	/*
	 * @return true iff the buffer is shuffled
	 */
	bool shuffleBuffer() const;

	/*
	 * @return the total number of features in the cache
	 */
	u32 totalNumberOfFeatures() const;

	/*
	 * @return the dimension of a feature vector
	 */
	u32 featureDimension() const;

	/*
	 * resets the buffer such that all features can again be accessed via next()
	 */
	virtual void newEpoch();
};

/**
 * HeaderReader
 *
 * reads only the header of the cache
 */
class HeaderReader : public BaseFeatureReader
{
private:
	typedef BaseFeatureReader Precursor;
protected:
	virtual void bufferNext() {}
	virtual bool isSequenceReader() { return false; }
public:
	HeaderReader(const char* name = "features.feature-reader");
	virtual ~HeaderReader() {}
	virtual void initialize();
};

/**
 * FeatureReader
 *
 * reads (unlabeled) feature vectors
 */
class FeatureReader : public BaseFeatureReader
{
private:
	typedef BaseFeatureReader Precursor;
protected:
	std::vector< Math::Vector<Float> > buffer_;	// the feature buffer
	bool needsContext_;
	Math::Matrix<Float> prebufferedSequence_;
	u32 prebufferPointer_;
	virtual bool isSequenceReader() { return false; }
	virtual void bufferNext();
public:
	FeatureReader(const char* name = "features.feature-reader");
	virtual ~FeatureReader() {}

	virtual void initialize();

	/*
	 * @return true iff there is at least one feature vector that has not been accessed via next()
	 */
	bool hasFeatures() const;

	/*
	 * @return the next feature vector to be processed (with index currentFeatureIndex_)
	 */
	const Math::Vector<Float>& next();
};

/**
 * SequenceFeatureReader
 *
 * reads (unlabeled) sequences of feature vectors
 *
 * nBufferedFeatures, ... now refers to sequences of features
 */
class SequenceFeatureReader : public BaseFeatureReader
{
private:
	typedef BaseFeatureReader Precursor;
	static const Core::ParameterBool paramSortSequences_;
private:
	/* helper struct for sorting the sequences according to their length */
	struct SequenceLengthComparator {
		SequenceFeatureReader* s_;
		SequenceLengthComparator(SequenceFeatureReader* s) : s_(s) {}
		bool operator() (u32 i, u32 j) {
			require_le(s_->nBufferedFeatures_, s_->buffer_.size());
			return (s_->buffer_.at(i).nColumns() > s_->buffer_.at(j).nColumns());
		}
	};
protected:
	std::vector< Math::Matrix<Float> > buffer_;	// the feature buffer
	std::vector< std::vector<u32> > timestamps_; // contains the time stamps of each feature vector from the buffered sequences
	u32 currentSequenceLength_;					// number of feature vectors in the current sequence
	bool sortSequences_;						// sort sequences according to length in descending order if true

	virtual void fillBuffer();
	virtual bool isSequenceReader() { return true; }
	virtual void bufferNext();
	void sortSequences();
public:
	SequenceFeatureReader(const char* name = "features.feature-reader");
	virtual ~SequenceFeatureReader() {}

	virtual void initialize();

	/*
	 * @return the total number of sequences in the cache
	 */
	u32 totalNumberOfSequences() const;

	/*
	 * @return true iff there is at least one feature sequence that has not been accessed via next()
	 */
	bool hasSequences() const;

	/*
	 * @return true iff the sequences in the buffer are sorted by their length in descending order
	 */
	bool areSequencesSorted() const;

	/*
	 * resets the buffer such that all feature sequences can again be accessed via next()
	 */
	virtual void newEpoch();

	std::vector<u32>& currentTimestamps();

	/*
	 * @return the next feature sequence to be processed (with index currentFeatureIndex_)
	 */
	const Math::Matrix<Float>& next();
};

/**
 * LabelReader
 *
 * reads a label cache
 */
class LabelReader : public FeatureReader
{
private:
	typedef FeatureReader Precursor;
protected:
	virtual void validateCacheFormat();
public:
	LabelReader(const char* name = "features.label-reader") : Precursor(name) {}
	virtual ~LabelReader() {}

	virtual void initialize();

	/*
	 * @return the next label
	 */
	u32 next();
};

/**
 * SequenceLabelReader
 *
 * reads a sequence label cache
 */
class SequenceLabelReader : public SequenceFeatureReader
{
private:
	typedef SequenceFeatureReader Precursor;
protected:
	std::vector<u32> labels_;
	virtual void validateCacheFormat();
public:
	SequenceLabelReader(const char* name = "features.label-reader") : Precursor(name) {}
	virtual ~SequenceLabelReader() {}

	virtual void initialize();

	/*
	 * @return the next label
	 */
	std::vector<u32>& next();
};

} // namespace


#endif /* FEATURES_FEATUREREADER_HH_ */
