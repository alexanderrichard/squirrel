#include "FeatureWriter.hh"
#include <stdio.h>

using namespace Features;

/*
 * FeatureWriter
 */

const Core::ParameterString BaseFeatureWriter::paramFeatureCacheFile_("feature-cache", "", "features.feature-writer");

BaseFeatureWriter::BaseFeatureWriter(const char* name, FeatureCache::CacheFormat cacheFormat) :
		name_(name),
		cacheFilename_(Core::Configuration::config(paramFeatureCacheFile_, name_)),
		cacheFormat_(cacheFormat),
		nWrittenFeatures_(0),
		featureDim_(0),
		isInitialized_(false),
		isFinalized_(false)
{}

BaseFeatureWriter::~BaseFeatureWriter() {
	if (!isFinalized_ && (nWrittenFeatures_ > 0)) {
		finalize();
	}
}

void BaseFeatureWriter::writeHeader(FeatureCache::FeatureType featureType) {
	require(isInitialized_);
	u8 type = ( featureType == FeatureCache::single ? 0 : 1 );
	cache_ << '#';
	switch (cacheFormat_) {
	case FeatureCache::labelCache: cache_ << 'l'; break;
	case FeatureCache::featureCache: cache_ << 'c'; break;
	default: cache_ << '?';
	}
	cache_ << (u8)1 << (u8)type << (u32)0 << (u32)featureDim_;
	if (featureType == FeatureCache::sequence) {
		cache_ << (u32) 0;
	}
}

void BaseFeatureWriter::updateHeader() {
	require(!cache_.is_open());
	FILE* pf;
	pf = fopen(cacheFilename_.c_str(), "r+w");
	if (!pf) {
		std::cerr << "FeatureWriter::updateHeader(): Could not open file " << cacheFilename_ << ". Abort." << std::endl;
		exit(1);
	}
	fseek(pf, 4, SEEK_SET); // skip first four bytes of header
	fwrite(&nWrittenFeatures_, sizeof(u32), 1, pf);
	fclose(pf);
}

void BaseFeatureWriter::initialize(u32 featureDim) {
	featureDim_ = featureDim;
	require(!cacheFilename_.empty());
	cache_.open(cacheFilename_, std::ios::out);
	isInitialized_ = true;
}

void BaseFeatureWriter::setCacheFilename(const std::string& filename) {
	require(!isInitialized_);
	cacheFilename_ = filename;
}

const std::string& BaseFeatureWriter::getCacheFilename() {
	return cacheFilename_;
}

void BaseFeatureWriter::finalize() {
	require(nWrittenFeatures_ > 0);
	if (cache_.is_open()) {
		cache_.close();
	}
	updateHeader();
	Core::Log::openTag("features.feature-writer");
	Core::Log::os("Wrote cache ") << cacheFilename_;
	Core::Log::closeTag();
	isFinalized_ = true;
}

/*
 * SingleFeatureWriter
 */

FeatureWriter::FeatureWriter(const char* name, FeatureCache::CacheFormat cacheFormat) :
		Precursor(name, cacheFormat)
{}

void FeatureWriter::write(const Math::Vector<Float>& f) {
	if (!isInitialized_) {
		initialize(f.nRows());
		writeHeader(FeatureCache::single);
	}
	require(cache_.is_open());
	for (u32 i = 0; i < f.nRows(); i++) {
		cache_ << (f32)f.at(i);
	}
	nWrittenFeatures_++;
}

void FeatureWriter::write(const Math::Matrix<Float>& featureSequence) {
	if (!isInitialized_) {
		initialize(featureSequence.nRows());
		writeHeader(FeatureCache::single);
	}
	require(cache_.is_open());
	for (u32 col = 0; col < featureSequence.nColumns(); col++) {
		for (u32 row = 0; row < featureSequence.nRows(); row++) {
			cache_ << (f32)featureSequence.at(row, col);
		}
	}
	nWrittenFeatures_ += featureSequence.nColumns();
}

/*
 * SequenceFeatureWriter
 */

SequenceFeatureWriter::SequenceFeatureWriter(const char* name, FeatureCache::CacheFormat cacheFormat) :
		Precursor(name, cacheFormat),
		nSequences_(0)
{}

SequenceFeatureWriter::~SequenceFeatureWriter() {
	if (!isFinalized_ && (nWrittenFeatures_ > 0)) {
		finalize();
	}
}

void SequenceFeatureWriter::updateNumberOfSequences() {
	require(!cache_.is_open());
	FILE* pf;
	pf = fopen(cacheFilename_.c_str(), "r+w");
	if (!pf) {
		std::cerr << "FeatureWriter::updateHeader(): Could not open file " << cacheFilename_ << ". Abort." << std::endl;
		exit(1);
	}
	fseek(pf, 12, SEEK_SET); // skip first twelve bytes of header
	fwrite(&nSequences_, sizeof(u32), 1, pf);
	fclose(pf);
}

void SequenceFeatureWriter::write(std::vector<u32>& timestamps, const Math::Matrix<Float>& featureSequence) {
	if (!isInitialized_) {
		initialize(featureSequence.nRows());
		writeHeader(FeatureCache::sequence);
	}
	require(cache_.is_open());
	require_eq(timestamps.size(), featureSequence.nColumns());
	cache_ << featureSequence.nColumns(); // sequence length
	for (u32 col = 0; col < featureSequence.nColumns(); col++) {
		cache_ << timestamps.at(col);
		for (u32 row = 0; row < featureSequence.nRows(); row++) {
			cache_ << (f32)featureSequence.at(row, col);
		}
	}
	nWrittenFeatures_ += featureSequence.nColumns();
	nSequences_++;
}

void SequenceFeatureWriter::write(const Math::Matrix<Float>& featureSequence) {
	std::vector<u32> timestamps(featureSequence.nColumns(), 0);
	for (u32 i = 0; i < timestamps.size(); i++)
		timestamps.at(i) = i;
	write(timestamps, featureSequence);
}

void SequenceFeatureWriter::finalize() {
	Precursor::finalize();
	updateNumberOfSequences();
}
