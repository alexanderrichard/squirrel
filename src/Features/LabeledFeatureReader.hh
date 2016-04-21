#ifndef FEATURES_LABELEDFEATUREREADER_HH_
#define FEATURES_LABELEDFEATUREREADER_HH_

#include "Core/CommonHeaders.hh"
#include "FeatureReader.hh"
#include "Math/Vector.hh"
#include "Math/Matrix.hh"
#include "FeatureReader.hh"

namespace Features {

/**
 * BaseLabeledFeatureReader
 */
class BaseLabeledFeatureReader
{
private:
	static const Core::ParameterString paramLabelCache_;
protected:
	std::string labelCacheFile_;
	FeatureCache labelCache_;
	virtual bool checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache);
public:
	BaseLabeledFeatureReader(const char* name);
	virtual ~BaseLabeledFeatureReader() {}
	virtual void initialize();
};

/**
 * LabeledFeatureReader
 *
 * non-sequential/non-temporal features caches, provides one label per feature vector
 */
class LabeledFeatureReader : public FeatureReader, BaseLabeledFeatureReader
{
private:
	typedef FeatureReader Precursor;
private:
	std::vector<u32> labelBuffer_;	// the label buffer
	u32 label_;

	virtual void bufferNext();
	virtual bool checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache);
public:
	LabeledFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~LabeledFeatureReader() {}

	virtual void initialize();
	/*
	 * start a new epoch, same as in FeatureReader
	 */
	virtual void newEpoch();

	/*
	 * @return the corresponding label
	 */
	u32 label();
};

/**
 * LabeledSequenceFeatureReader
 *
 * provides ONE label per sequence
 * (use TemporallyLabeledSequenceFeatureReader if one label for each feature vector of the sequence is required)
 */
class LabeledSequenceFeatureReader : public SequenceFeatureReader, BaseLabeledFeatureReader
{
private:
	typedef SequenceFeatureReader Precursor;
private:
	std::vector<u32> labelBuffer_;	// the label buffer

	virtual void bufferNext();
	virtual bool checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache);
public:
	LabeledSequenceFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~LabeledSequenceFeatureReader() {}

	virtual void initialize();
	virtual void newEpoch();

	/*
	 * @return the leading label for this sequence (label of first feature in the sequence)
	 */
	u32 label();
};

/**
 * TemporallyLabeledSequenceFeatureReader
 *
 * requires label cache to provide a label for each feature vector within the sequence
 */
class TemporallyLabeledSequenceFeatureReader : public SequenceFeatureReader, BaseLabeledFeatureReader
{
private:
	typedef SequenceFeatureReader Precursor;
private:
	std::vector< std::vector<u32> > labelBuffer_;	// the label buffer

	virtual void bufferNext();
	virtual bool checkConsistency(FeatureCache& featureCache, FeatureCache& labelCache);
public:
	TemporallyLabeledSequenceFeatureReader(const char* name = "features.labeled-feature-reader");
	virtual ~TemporallyLabeledSequenceFeatureReader() {}

	virtual void initialize();
	/*
	 * start a new epoch, same as in SequenceFeatureReader
	 */
	virtual void newEpoch();

	/*
	 * @return the corresponding labels for each feature vector in the sequence
	 */
	const std::vector<u32>& labelSequence();
};

} // namespace

#endif /* LABELEDFEATUREREADER_HH_ */
