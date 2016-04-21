#ifndef FEATURES_FEATUREWRITER_HH_
#define FEATURES_FEATUREWRITER_HH_

#include "Core/CommonHeaders.hh"
#include "Math/Vector.hh"
#include "FeatureCache.hh"
#include <string.h>
#include <vector>

namespace Features {

/*
 * feature writer base class
 */
class BaseFeatureWriter
{
private:
	static const Core::ParameterString paramFeatureCacheFile_;
protected:
	const char* name_;

	Core::BinaryStream cache_;
	std::string cacheFilename_;
	FeatureCache::CacheFormat cacheFormat_;
	u32 nWrittenFeatures_;
	u32 featureDim_;
	bool isInitialized_;
	bool isFinalized_;

	void writeHeader(FeatureCache::FeatureType featureType);
	virtual void updateHeader();
	void initialize(u32 featureDim);
public:
	BaseFeatureWriter(const char* name, FeatureCache::CacheFormat cacheFormat);
	virtual ~BaseFeatureWriter();
	// may be helpful, but in general use the parameter paramFeatureCacheFile_
	void setCacheFilename(const std::string& filename);
	const std::string& getCacheFilename();
	virtual void write(const Math::Matrix<Float>& featureSequence) = 0;
	virtual u32 nWrittenFeatures() { return nWrittenFeatures_; }
	virtual void finalize();
};

/*
 * single feature writer
 */
class FeatureWriter : public BaseFeatureWriter
{
private:
	typedef BaseFeatureWriter Precursor;
public:
	FeatureWriter(const char* name = "features.feature-writer", FeatureCache::CacheFormat cacheFormat = FeatureCache::featureCache);
	virtual ~FeatureWriter() {}
	/*
	 * @param featureSequence all feature vectors in this sequence will be written to the output feature cache
	 */
	virtual void write(const Math::Matrix<Float>& featureSequence);
	/*
	 * @param feature this feature vector will be written to the output feature cache
	 */
	virtual void write(const Math::Vector<Float>& feature);
};

/*
 * sequence feature writer
 */
class SequenceFeatureWriter : public BaseFeatureWriter
{
private:
	typedef BaseFeatureWriter Precursor;
private:
	u32 nSequences_;
	void updateNumberOfSequences();
public:
	SequenceFeatureWriter(const char* name = "features.feature-writer", FeatureCache::CacheFormat cacheFormat = FeatureCache::featureCache);
	virtual ~SequenceFeatureWriter();
	/*
	 * @timestamps the timestamp for each feature vector in the feature sequence
	 * @featureSequence the sequence to be written to the output feature cache
	 */
	virtual void write(std::vector<u32>& timestamps, const Math::Matrix<Float>& featureSequence);
	/*
	 * @featureSequence the sequence to be written to the output feature cache
	 * timestamps are generated automatically, starting from 0 and increasing by 1 for each feature vector
	 */
	virtual void write(const Math::Matrix<Float>& featureSequence);
	u32 nWrittenSequences() { return nSequences_; }
	virtual void finalize();
};

} // namespace

#endif /* FEATURES_FEATUREWRITER_HH_ */
