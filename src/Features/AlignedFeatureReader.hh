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
 * LabeledFeatureReader.hh
 *
 *  Created on: Apr 11, 2014
 *      Author: richard
 */

#ifndef FEATURES_LABELEDFEATUREREADER_HH_
#define FEATURES_LABELEDFEATUREREADER_HH_

#include "Core/CommonHeaders.hh"
#include "FeatureReader.hh"
#include "Math/Vector.hh"
#include "Math/Matrix.hh"
#include "FeatureReader.hh"

namespace Features {

class BaseAlignedFeatureReader
{
private:
	static const Core::ParameterString paramLabelCache_; // deprecated, just for compatibility reasons
	static const Core::ParameterString paramTargetCache_;
protected:
	std::string targetCacheFile_;
public:
	BaseAlignedFeatureReader(const char* name);
	virtual ~BaseAlignedFeatureReader() {}
};

/**
 * AlignedFeatureReader
 *
 * non-sequential/non-temporal features caches, provides one target vector per feature vector
 */
class AlignedFeatureReader : public FeatureReader, BaseAlignedFeatureReader
{
private:
	typedef FeatureReader Precursor;
protected:
	FeatureReader targetReader_;
	Math::Vector<Float> target_;

	virtual void shuffleIndices();
public:
	AlignedFeatureReader(const char* name = "features.aligned-feature-reader");
	virtual ~AlignedFeatureReader() {}
	virtual void initialize();
	virtual void newEpoch();
	virtual const Math::Vector<Float>& next();

	/*
	 * @return dimension of target vector
	 */
	virtual u32 targetDimension() const;
	/*
	 * @return the corresponding target vector
	 */
	const Math::Vector<Float>& target() const;
};

/**
 * LabeledFeatureReader
 *
 * non-sequential/non-temporal features caches, provides one label per feature vector
 */
class LabeledFeatureReader : public AlignedFeatureReader
{
private:
	typedef AlignedFeatureReader Precursor;
	const Math::Vector<Float>& target() const { return Precursor::target(); }
public:
	LabeledFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~LabeledFeatureReader() {}
	virtual void initialize();

	/*
	 * @return the corresponding label
	 */
	u32 label() const;

	/*
	 * @return the number of classes
	 */
	u32 nClasses() const;
};

/**
 * AlignedSequenceFeatureReader
 *
 * provides ONE target vector per sequence
 */
class AlignedSequenceFeatureReader : public SequenceFeatureReader, BaseAlignedFeatureReader
{
private:
	typedef SequenceFeatureReader Precursor;
protected:
	FeatureReader targetReader_;
	Math::Vector<Float> target_;

	virtual void shuffleIndices();
	virtual void sortSequences();
public:
	AlignedSequenceFeatureReader(const char* name = "features.aligned-feature-reader");
	virtual ~AlignedSequenceFeatureReader() {}
	virtual void initialize();
	virtual void newEpoch();
	virtual const Math::Matrix<Float>& next();

	/*
	 * @return dimension of target vector
	 */
	virtual u32 targetDimension() const;
	/*
	 * @return the corresponding target vector
	 */
	const Math::Vector<Float>& target() const;
};

/**
 * LabeledSequenceFeatureReader
 *
 * provides ONE label per sequence
 * (use TemporallyLabeledSequenceFeatureReader if one label for each feature vector of the sequence is required)
 */
class LabeledSequenceFeatureReader : public AlignedSequenceFeatureReader
{
private:
	typedef AlignedSequenceFeatureReader Precursor;
	const Math::Vector<Float>& target() const { return Precursor::target(); }
public:
	LabeledSequenceFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~LabeledSequenceFeatureReader() {}
	virtual void initialize();

	/*
	 * @return the corresponding label
	 */
	u32 label() const;

	/*
	 * @return the number of classes
	 */
	u32 nClasses() const;
};

/**
 * TemporallyAlignedSequenceFeatureReader
 *
 * requires target cache to provide a target vector for each feature vector within the sequence
 */
class TemporallyAlignedSequenceFeatureReader : public SequenceFeatureReader, BaseAlignedFeatureReader
{
private:
	typedef SequenceFeatureReader Precursor;
protected:
	SequenceFeatureReader targetReader_;
	Math::Matrix<Float> target_;

	virtual void shuffleIndices();
	virtual void sortSequences();
public:
	TemporallyAlignedSequenceFeatureReader(const char* name = "features.aligned-feature-reader");
	virtual ~TemporallyAlignedSequenceFeatureReader() {}
	virtual void initialize();
	virtual void newEpoch();
	virtual const Math::Matrix<Float>& next();

	/*
	 * @return dimension of target vector
	 */
	virtual u32 targetDimension() const;
	/*
	 * @return the corresponding target matrix
	 */
	const Math::Matrix<Float>& target() const;
};

/**
 * TemporallyLabeledSequenceFeatureReader
 *
 * requires target cache to provide a target vector for each feature vector within the sequence
 */
class TemporallyLabeledSequenceFeatureReader : public TemporallyAlignedSequenceFeatureReader
{
private:
	typedef TemporallyAlignedSequenceFeatureReader Precursor;
	const Math::Matrix<Float>& target() const { return Precursor::target(); }
	std::vector<u32> labelSequence_;
public:
	TemporallyLabeledSequenceFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~TemporallyLabeledSequenceFeatureReader() {}
	virtual void initialize();
	virtual const Math::Matrix<Float>& next();

	/*
	 * @return the corresponding label sequence
	 */
	const std::vector<u32>& labelSequence() const;

	/*
	 * @return the number of classes
	 */
	u32 nClasses() const;
};

} // namespace

#endif /* LABELEDFEATUREREADER_HH_ */
