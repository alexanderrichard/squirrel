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
 * FeatureCache.hh
 *
 *  Created on: Apr 4, 2014
 *      Author: richard
 */

#ifndef FEATURES_FEATURECACHE_HH_
#define FEATURES_FEATURECACHE_HH_

#include <Modules.hh>
#include <vector>
#include <string.h>
#include <fstream>
#include "Core/CommonHeaders.hh"
#include "Math/Vector.hh"
#include <sstream>
#include <sys/stat.h>

#ifdef MODULE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else
namespace cv {
	typedef u32 Mat; // dummy definition if OpenCV is not used
}
#endif

namespace Features {


/*
 * FeatureCache
 */
class FeatureCache
{
private:
	static const Core::ParameterBool paramLogCacheInformation_;
	static const Core::ParameterFloat paramRawScale_;
public:
	enum FeatureType { none = 0, vectors = 1, sequences = 2, images = 3, videos = 4, labels = 5, sequencelabels = 6};
private:
	// struct holds information about a cache file
	struct CacheSpecifier {
		std::vector<std::string> cacheFilename; // usually only one entry, only for videos multiple image filenames
		u32 cacheSize;
		u32 nSequences;
	};

	std::vector<CacheSpecifier> caches_;
	u32 currentCacheIndex_;			// index of the current cache in caches_
	Core::IOStream *cacheFile_;

	FeatureType featureType_;
	u32 featureDim_;

	u32 cacheSize_;					// number of features in the cache (for bundle files: overall number in all sub-caches)
	// information about sequences (if featureType_ == sequence)
	u32 nSequences_;				// number of sequences in the cache (for bundle files: overall number in all sub-caches)

	bool isInitialized_;
	bool logCacheInformation_;

	u32 width_;
	u32 height_;
	u32 channels_;

	Math::Matrix<f32> inputBuffer_;

	Float rawScale_;
	static FeatureType getType(Core::IOStream* stream);

	void convertImageToVector(cv::Mat& image, u32 column = 0);
	void convertStringToVector(std::string& str, u32 column = 0);
	void readVector();
	void readSequence();
	void readVideo(const std::vector<std::string>& videoFrames);
	void readImage(const std::string& imageFile, u32 column = 0);

	std::vector<u32> getCacheHeaderSpecifications();
	void fillInputBuffer();

	void validateCacheHeaders(const std::string& cacheFilename);
	void logCacheInformation(const std::string& cacheFilename);
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
	FeatureType featureType() const { require(isInitialized_); return featureType_; }
	u32 cacheSize() const { require(isInitialized_); return cacheSize_; }
	u32 featureDim() const { require(isInitialized_); return featureDim_; }
	u32 nSequences() const { require(isInitialized_); return nSequences_; }

	/**
	 * use this functions to access the data
	 * @return the next float/int in the cache(s) (header excluded)
	 */
	const Math::Matrix<Float>& next();

	/**
	 * return the FeatureType of the specified cache file
	 */
	static FeatureType featureType(const std::string& cachefile);
};

} // namespace


#endif /* FEATURES_FEATURECACHE_HH_ */
