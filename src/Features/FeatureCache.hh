#ifndef FEATURES_FEATURECACHE_HH_
#define FEATURES_FEATURECACHE_HH_

#include <vector>
#include <string.h>
#include <fstream>
#include "Core/CommonHeaders.hh"
#include "Math/Vector.hh"

namespace Features {

/**
 * FILE FORMAT FOR FEATURE CACHES
 *
 * features are always stored as f32, but are converted to f64 while reading if Core/Types.hh defines Float as f64
 *
 * header (see FeatureCache::readCacheSpecification)
 *
 * single features:
 * <feature1 dim1><feature1 dim2>...
 * <feature2 dim1><feature2 dim2>...
 * ...
 * sequence features:
 * <length of sequence1>
 * <sequence1-feature1 time stamp><feature1 dim1><feature1 dim2>...
 * <sequence1feature2 time stamp><feature2 dim1><feature2 dim2>...
 * ...
 * <length of sequence2>
 * <sequence12-feature1 time stamp><feature1 dim1><feature1 dim2>...
 * <sequence2feature2 time stamp><feature2 dim1><feature2 dim2>...
 * ...
 *
 * CACHE BUNDLES
 *
 * bundle file must not contain leading spaces or empty lines
 *
 * header (must be the first line of the file)
 * #bundle
 *
 * multiple caches specified in a bundle file can be read and are then treated as a single large cache
 * bundle file format is an Ascii file containing a list of cache files:
 * <path-to-subcache1>
 * <path-to-subcache2>
 * ...
 */

/**
 * FeatureCache
 *
 * A feature reader should create an instance of this class and access the cache file(s)
 * via read_Float and the cache information methods.
 *
 * The class shadows the fact that there might be multiple cache files due to bundle files.
 * Multiple caches are read sequentially, giving the appearance that there is only a single cache.
 */
class FeatureCache
{
private:
	static const Core::ParameterBool paramLogCacheInformation_;
public:
	enum CacheFormat {featureCache = 0, labelCache = 1};
	enum FeatureType {single = 0, sequence = 1};
private:
	// struct holds information about a cache file
	struct CacheSpecifier {
		std::string cacheFilename;
		CacheFormat cacheFormat;
		u8 version;
		FeatureType featureType;
		u32 cacheSize;
		u32 featureDim;
		u32 nSequences;
		u64 nRemainingFloats;
	};
	std::vector<CacheSpecifier> caches_;
	u32 currentCacheIndex_;			// index of the current cache in caches_
	Core::BinaryStream cacheFile_;

	u64 distanceToNextSequenceLength_;	// contains number of floats to read before next segment length (for sequence features)

	u32 cacheSize_;					// number of features in the cache (for bundle files: overall number in all sub-caches)
	// information about sequences (if featureType_ == sequence)
	u32 nSequences_;				// number of sequences in the cache (for bundle files: overall number in all sub-caches)

	bool isInitialized_;
	bool logCacheInformation_;

	bool isBundleFile(const std::string& filename);
	void validateCacheHeaders(const std::string& cacheFilename);
	bool readCacheSpecification(Core::BinaryStream& stream, CacheSpecifier& cacheSpec);
	void logCacheInformation(const std::string& cacheFilename);
	f32 read_f32();
public:
	FeatureCache();
	virtual ~FeatureCache() {}
	virtual void initialize(const std::string& cacheFilename);

	void setLogCacheInformation(bool logCacheInformation);

	/*
	 * reset to the state after initialization
	 */
	void reset();

	/**
	 * return cache information (see FeatureCache::readCacheSpecification)
	 */
	CacheFormat cacheFormat() const { require(caches_.size() > 0); return caches_[0].cacheFormat; }
	u8 version() const { require(caches_.size() > 0); return caches_[0].version; }
	FeatureType featureType() const { require(caches_.size() > 0); return caches_[0].featureType; }
	u32 cacheSize() const { return cacheSize_; }
	u32 featureDim() const { require(caches_.size() > 0); return caches_[0].featureDim; }
	u32 nSequences() const { return nSequences_; }
	bool nextIsSequenceLengthEntry() const { return (distanceToNextSequenceLength_ == 0); }

	/**
	 * use this functions to access the data
	 * @return the next float/int in the cache(s) (header excluded)
	 */
	Float read_Float();
	u32 read_u32();
};

} // namespace


#endif /* FEATURES_FEATURECACHE_HH_ */
