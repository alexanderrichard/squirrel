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

#include "FeatureCache.hh"
#include <stdlib.h>

using namespace Features;

const Core::ParameterBool FeatureCache::paramLogCacheInformation_("log-cache-information", true, "features.feature-cache");

FeatureCache::FeatureCache() :
		currentCacheIndex_(0),
		distanceToNextSequenceLength_(0),
		cacheSize_(0),
		nSequences_(0),
		isInitialized_(false),
		logCacheInformation_(Core::Configuration::config(paramLogCacheInformation_))
{}

void FeatureCache::initialize(const std::string& cacheFilename) {
	// validate and store all cache headers (may be multiple in case of a bundle file)
	require(!cacheFilename.empty());
	validateCacheHeaders(cacheFilename);
	cacheFile_.open(caches_[currentCacheIndex_].cacheFilename, std::ios::in);
	// skip header
	CacheSpecifier dummy;
	readCacheSpecification(cacheFile_, dummy);
	if (logCacheInformation_) {
		logCacheInformation(cacheFilename);
	}
	isInitialized_ = true;
}

void FeatureCache::setLogCacheInformation(bool logCacheInformation) {
	logCacheInformation_ = logCacheInformation;
}

void FeatureCache::reset() {
	currentCacheIndex_ = 0;
	if (cacheFile_.is_open()) {
		cacheFile_.close();
	}
	for (u32 i = 0; i < caches_.size(); i++) {
		// total number of floats/u32 (header excluded) in cache:
		// featureDim * cacheSize + (cacheSize time stamps + nSequences sequence length indications) for sequence features
		caches_[i].nRemainingFloats = caches_[i].cacheSize * caches_[i].featureDim
				+ (caches_[i].featureType == sequence ? (caches_[i].cacheSize + caches_[i].nSequences) : 0);
	}
	// re-open first cache file
	cacheFile_.open(caches_[currentCacheIndex_].cacheFilename, std::ios::in);
	// skip header
	CacheSpecifier dummy;
	readCacheSpecification(cacheFile_, dummy);
}

bool FeatureCache::isBundleFile(const std::string& filename) {
	std::ifstream file(filename.c_str());
	std::string tmp;
	std::getline(file, tmp);
	file.close();
	if (tmp.compare(std::string("#bundle")) == 0)
		return true;
	else
		return false;
}

void FeatureCache::validateCacheHeaders(const std::string& cacheFilename) {
	// if bundle file: combine headers of all caches in bundle file and check validity
	if (isBundleFile(cacheFilename)) {
		std::ifstream bundle(cacheFilename.c_str());
		std::string line;
		// skip header of bundle file
		std::getline(bundle, line);
		// read header of each specified cache
		while (std::getline(bundle, line)) {
			Core::BinaryStream stream(line, std::ios::in);
			caches_.push_back(CacheSpecifier());
			caches_.back().cacheFilename = std::string(line);
			if (!readCacheSpecification(stream, caches_.back())) {
				std::cerr << "FeatureCache: Header of cache " << line
						<< " not valid or incompatible with other caches from " << cacheFilename << ". Abort." << std::endl;
				exit(1);
			}
			stream.close();
			// check header fields for consistency
			require(caches_[0].cacheFormat == caches_.back().cacheFormat);
			require(caches_[0].version == caches_.back().version);
			require(caches_[0].featureType == caches_.back().featureType);
			require(caches_[0].featureDim == caches_.back().featureDim);
		}
		bundle.close();
	}
	// if not bundle file: just read header of given cache file
	else {
		Core::BinaryStream stream(cacheFilename, std::ios::in);
		caches_.push_back(CacheSpecifier());
		caches_.back().cacheFilename = std::string(cacheFilename);
		if (!readCacheSpecification(stream, caches_.back())) {
			std::cerr << "FeatureCache: Header of cache " << cacheFilename << " not valid. Abort." << std::endl;
			exit(1);
		}
		stream.close();
	}
	// update "global" cache variables
	for (u32 i = 0; i < caches_.size(); i++) {
		cacheSize_ += caches_[i].cacheSize;
		nSequences_ += caches_[i].nSequences;
	}
}

bool FeatureCache::readCacheSpecification(Core::BinaryStream& stream, CacheSpecifier& cacheSpec) {
	/*
	 * header:
	 * '#' as first symbol
	 * cache identifier (char, "c" for feature caches, "l" for label caches)
	 * cache version (u8)
	 * feature type (u8, 0 for single, 1 for sequence)
	 * total number of feature vectors in cache (u32)
	 * feature vector dimension (u32)
	 * if feature type 1 (sequence):
	 *   number of sequences in cache (u32)
	 */
	require(stream.is_open());
	// read header fields
	char ch;
	stream >> ch;
	if (ch != '#')
		return false;
	stream >> ch;
	switch (ch) {
	case 'c': cacheSpec.cacheFormat = featureCache; break;
	case 'l': cacheSpec.cacheFormat = labelCache; break;
	default: return false;
	}
	stream >> cacheSpec.version;
	if (cacheSpec.version != 1) // current version is 1
		return false;
	u8 fType;
	stream >> fType;
	switch (fType) {
	case 0: cacheSpec.featureType = single; break;
	case 1: cacheSpec.featureType = sequence; break;
	default: return false;
	}
	stream >> cacheSpec.cacheSize;
	stream >> cacheSpec.featureDim;
	// sequence information is only contained in the header if featureType_ == sequence
	if (cacheSpec.featureType == sequence) {
		stream >> cacheSpec.nSequences;
	}
	else {
		cacheSpec.nSequences = 0;
	}
	// total number of floats (header excluded) in cache:
	// featureDim * cacheSize + (cacheSize time stamps + nSequences sequence length indications) for sequence features
	cacheSpec.nRemainingFloats = (u64)cacheSpec.cacheSize * (u64)cacheSpec.featureDim
			+ (cacheSpec.featureType == sequence ? ((u64)cacheSpec.cacheSize + (u64)cacheSpec.nSequences) : 0);
	return true;
}

f32 FeatureCache::read_f32() {
	require(isInitialized_);
	require(cacheFile_.is_open());
	// if no floats remaining in this cache, open next cache
	if (caches_[currentCacheIndex_].nRemainingFloats == 0) {
		currentCacheIndex_++;
		require_lt(currentCacheIndex_, caches_.size());
		cacheFile_.close();
		cacheFile_.open(caches_[currentCacheIndex_].cacheFilename, std::ios::in);
		// skip header
		CacheSpecifier dummy;
		readCacheSpecification(cacheFile_, dummy);
	}
	require(caches_[currentCacheIndex_].nRemainingFloats > 0);
	// one float is read in this function
	f32 data;
	cacheFile_ >> data;
	caches_[currentCacheIndex_].nRemainingFloats--;
	if (featureType() == sequence) {
		// check if this field is a sequence length field
		if (distanceToNextSequenceLength_ == 0) {
			u32 tmp = reinterpret_cast<u32 &> (data);
			distanceToNextSequenceLength_ = (u64)tmp;
			distanceToNextSequenceLength_ *= (u64)featureDim() + 1; // +1 due to time stamp
		}
		else {
			distanceToNextSequenceLength_--;
		}
	}
	return data;
}

Float FeatureCache::read_Float() {
	return (Float)read_f32();
}

u32 FeatureCache::read_u32() {
	f32 data = read_f32();
	return reinterpret_cast<u32 &> (data);
}

void FeatureCache::logCacheInformation(const std::string& cacheFilename) {
	Core::Log::openTag("feature-cache.information", cacheFilename.c_str());
	switch (cacheFormat()) {
	case featureCache: Core::Log::os("cache format: feature cache"); break;
	case labelCache: Core::Log::os("cache format: label cache"); break;
	}
	Core::Log::os("cache version: ") << (u32)version();
	if (featureType() == single)
		Core::Log::os("feature type: single");
	else
		Core::Log::os("feature type: sequence");
	Core::Log::os("total number of feature vectors: ") << cacheSize();
	Core::Log::os("feature vector dimension: ") << featureDim();
	if (featureType() == sequence)
		Core::Log::os("number of feature sequences: ") << nSequences();
	Core::Log::closeTag();
}
