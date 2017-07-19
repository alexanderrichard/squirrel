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
 * FeatureCache.cc
 *
 *  Created on: Apr 4, 2014
 *      Author: richard
 */

#include "FeatureCache.hh"
#include <stdlib.h>

using namespace Features;

const Core::ParameterBool FeatureCache::paramLogCacheInformation_("log-cache-information", true, "features.feature-cache");

const Core::ParameterFloat FeatureCache::paramRawScale_("raw-scale", 1.0, "features.feature-cache"); // set to 1.0/255.0 if images should be in range [0,1]

FeatureCache::FeatureCache() :
		currentCacheIndex_(0),
		cacheFile_(0),
		featureType_(none),
		featureDim_(0),
		cacheSize_(0),
		nSequences_(0),
		isInitialized_(false),
		logCacheInformation_(Core::Configuration::config(paramLogCacheInformation_)),
		width_(0),
		height_(0),
		channels_(0),
		inputBuffer_(0),
		rawScale_(Core::Configuration::config(paramRawScale_))
{}

void FeatureCache::initialize(const std::string& cacheFilename) {
	require(!cacheFilename.empty());
	validateCacheHeaders(cacheFilename);

	if (logCacheInformation_) {
		logCacheInformation(cacheFilename);
	}
	isInitialized_ = true;
}

void FeatureCache::setLogCacheInformation(bool logCacheInformation) {
	logCacheInformation_ = logCacheInformation;
}

void FeatureCache::convertImageToVector(cv::Mat& image, u32 column) {
#ifdef MODULE_OPENCV
	/* equalize width and height */
	if((u32)image.rows != height_ || (u32)image.cols != width_) {
		cv::Mat temp;
		cv::resize(image, temp, cv::Size(width_, height_));
		image = temp;
	}
	/* equalize channels */
	if((u32)image.channels() != channels_) {
		cv::Mat temp;
		if(image.channels() == 3 && channels_ == 1)
			cv::cvtColor(image, temp, CV_BGR2GRAY);
		else if(image.channels() == 1 && channels_ == 3)
			cv::cvtColor(image, temp, CV_GRAY2BGR);
		image = temp;
	}
	/* convert image to float image */
	cv::Mat temp;
	if(channels_ == 1)
		image.convertTo(temp, CV_32FC1, rawScale_);
	else
		image.convertTo(temp, CV_32FC3, rawScale_);
	image = temp;
	/* write to buffer */
	Core::Utils::copyCVMatToMemory(image, inputBuffer_.begin() + column * featureDim_);
#else
	Core::Error::msg("FeatureCache::convertImageToVector requires OpenCV but binary is not compiled with OpenCV support.") << Core::Error::abort;
#endif
}

void FeatureCache::convertStringToVector(std::string& str, u32 column) {
	if ((featureType_ == labels) || (featureType_ == sequencelabels)) {
		u32 label = atoi(str.c_str());
		if (label >= featureDim_)
			Core::Error::msg("FeatureCache::convertStringToVector: label index is ") << label << " but maximum allowed is " << featureDim_ - 1 << "." << Core::Error::abort;
		inputBuffer_.fill(0, column, featureDim_-1, column, 0);
		inputBuffer_.at(label, column) = 1.0;
	}
	else if ((featureType_ == vectors) || (featureType_ == sequences)) {
		std::vector< std::string > tokenized;
		Core::Utils::tokenizeString(tokenized, str);
		if (tokenized.size() != featureDim_)
			Core::Error::msg("FeatureCache::convertStringToVector: feature dimension mismatch (") << tokenized.size() << " vs. " << featureDim_ << ")" << Core::Error::abort;
		for (u32 d = 0; d < tokenized.size(); d++)
			inputBuffer_.at(d, column) = atof(tokenized.at(d).c_str());
	}
}

void FeatureCache::fillInputBuffer() {
	require_lt(currentCacheIndex_, caches_.size());
	switch(featureType_) {
	case vectors:
	case labels:
		readVector();
		break;
	case sequences:
	case sequencelabels:
		readSequence();
		break;
	case images:
		inputBuffer_.resize(channels_ * width_ * height_, 1);
		readImage(caches_[currentCacheIndex_].cacheFilename.back());
		currentCacheIndex_++;
		break;
	case videos:
		readVideo(caches_[currentCacheIndex_].cacheFilename);
		currentCacheIndex_++;
		break;
	default:
		// if we are here something went wrong
		Core::Error::msg("FeatureCache::openCache: Incorrect cache specification.") << Core::Error::abort;
	}
}

void FeatureCache::readVector() {
	// read binary file
	if (typeid(*cacheFile_) == typeid(Core::BinaryStream)) {
		inputBuffer_.resize(featureDim_, 1);
		// either labels...
		if ((featureType_ == labels)) {
			inputBuffer_.setToZero();
			u32 label;
			(*cacheFile_) >> label;
			inputBuffer_.at(label, 0) = 1.0;
		}
		// .. or feature vectors
		else {
			for (u32 d = 0; d < featureDim_; d++)
				(*cacheFile_) >> inputBuffer_.at(d, 0);
		}
	}
	// read ascii/gzipped file
	else {
		std::string line;
		if (!cacheFile_->getline(line))
			Core::Error::msg("FeatureCache::readVector: unexpected end of file.") << Core::Error::abort;
		inputBuffer_.resize(featureDim_, 1);
		convertStringToVector(line);
	}
}

void FeatureCache::readSequence() {
	// read binary file
	if (typeid(*cacheFile_) == typeid(Core::BinaryStream)) {
		u32 seqSize;
		(*cacheFile_) >> seqSize;
		inputBuffer_.resize(featureDim_, seqSize);
		// either sequence labels...
		if (featureType_ == sequencelabels) {
			inputBuffer_.setToZero();
			u32 label;
			for (u32 t = 0; t < seqSize; t++) {
				(*cacheFile_) >> label;
				inputBuffer_.at(label, t) = 1.0;
			}
		}
		// ... sequence of feature vectors
		else {
			for (u32 t = 0; t < seqSize; t++)
				for (u32 d = 0; d < featureDim_; d++)
					(*cacheFile_) >> inputBuffer_.at(d, t);
		}
	}
	// read ascii/gzipped file
	else {
		std::string line;
		std::vector<std::string> seq;
		while ((cacheFile_->getline(line)) && (line.compare("#") != 0)) {
			seq.push_back(line);
		}
		inputBuffer_.resize(featureDim_, seq.size());
		for (u32 i = 0; i < seq.size(); i++) {
			convertStringToVector(seq.at(i), i);
		}
	}
}

void FeatureCache::readImage(const std::string& imageFile, u32 column) {
#ifdef MODULE_OPENCV
	cv::Mat image = cv::imread(imageFile);
	if(image.data == NULL)
		Core::Error::msg() << "Unable to open Image: " << imageFile << Core::Error::abort;
	convertImageToVector(image, column);
#else
	Core::Error::msg("FeatureCache::readImage requires OpenCV but binary is not compiled with OpenCV support.") << Core::Error::abort;
#endif
}

void FeatureCache::readVideo(const std::vector<std::string>& videoFrames) {
	u32 nFrames = videoFrames.size();
	inputBuffer_.resize(channels_ * width_ * height_, nFrames);
	inputBuffer_.setToZero();
	for(u32 frameIdx = 0; frameIdx < nFrames; frameIdx++) {
		readImage(videoFrames.at(frameIdx), frameIdx);
	}
}

void FeatureCache::reset() {
	currentCacheIndex_ = 0;
	// re-open cache file
	if ((featureType_ == vectors) || (featureType_ == sequences) || (featureType_ == labels) || (featureType_ == sequencelabels)) {
		cacheFile_->close();
		cacheFile_->open(caches_[currentCacheIndex_].cacheFilename.back().c_str(), std::ios::in);
		// skip header
		std::string line;
		cacheFile_->getline(line);
		cacheFile_->getline(line);
	}
}

FeatureCache::FeatureType FeatureCache::getType(Core::IOStream* stream) {
	require(stream);
	std::string tmp;
	if (!stream->getline(tmp))
		Core::Error::msg("FeatureCache::getType: could not read cache file.") << Core::Error::abort;
	if (tmp.compare(std::string("#vectors")) == 0)
		return vectors;
	else if (tmp.compare(std::string("#sequences")) == 0)
		return sequences;
	else if (tmp.compare(std::string("#images")) == 0)
		return images;
	else if (tmp.compare(std::string("#videos")) == 0)
		return videos;
	else if (tmp.compare(std::string("#labels")) == 0)
		return labels;
	else if (tmp.compare(std::string("#sequencelabels")) == 0)
		return sequencelabels;
	else
		Core::Error::msg("FeatureCache::getType: Type must be one of #vectors, #sequences, #images, #videos, #labels, or #sequencelabels.") << Core::Error::abort;
	return none;
}

std::vector<u32> FeatureCache::getCacheHeaderSpecifications() {
	std::string line;
	// read second line of cache file to get cache specifications
	if (!cacheFile_->getline(line))
		Core::Error::msg("FeatureCache::getCacheHeaderSpecifications: could not read second line of cache file.") << Core::Error::abort;
	std::vector<std::string> tokens;
	Core::Utils::tokenizeString(tokens, line);
	std::vector<u32> result;
	for (u32 i = 0; i < tokens.size(); i++)
		result.push_back(atoi(tokens.at(i).c_str()));
	// sanity check
	if ((featureType_ == vectors) && (result.size() != 2))
		Core::Error::msg("Second row of cache header needs to be <total-number-of-features> <feature-dimension>.") << Core::Error::abort;
	if ((featureType_ == sequences) && (result.size() != 3))
		Core::Error::msg("Second row of cache header needs to be <total-number-of-features> <feature-dimension> <number-of-sequences>.") << Core::Error::abort;
	if (((featureType_ == images) || (featureType_ == videos)) && (result.size() != 3))
		Core::Error::msg("Second row of cache header needs to be <width> <height> <channels>.") << Core::Error::abort;
	if ((featureType_ == labels) && (result.size() != 2))
		Core::Error::msg("Second row of cache header needs to be <total-number-of-labels> <number-of-classes>.") << Core::Error::abort;
	if ((featureType_ == sequencelabels) && (result.size() != 3))
		Core::Error::msg("Second row of cache header needs to be <total-number-of-labels> <number-of-classes> <number-of-sequences>.") << Core::Error::abort;
	return result;
}

void FeatureCache::validateCacheHeaders(const std::string& cacheFilename) {
	// open file as IOStream
	if (Core::Utils::isGz(cacheFilename))
		cacheFile_ = new Core::CompressedStream(cacheFilename, std::ios::in);
	else if (Core::Utils::isBinary(cacheFilename))
		cacheFile_ = new Core::BinaryStream(cacheFilename, std::ios::in);
	else
		cacheFile_ = new Core::AsciiStream(cacheFilename, std::ios::in);
	featureType_ = getType(cacheFile_);
	std::vector<u32> headerSpecs = getCacheHeaderSpecifications();

	/* if ascii feature file: cache is the complete file */
	if ((featureType_ == vectors) || (featureType_ == sequences) || (featureType_ == labels) || (featureType_ == sequencelabels)) {
		// read header: total number of feature vectors, feature dimension, number of sequences (if sequence cache)
		caches_.push_back(CacheSpecifier());
		caches_.back().cacheFilename.push_back(cacheFilename);
		caches_.back().cacheSize = headerSpecs.at(0);
		featureDim_ = headerSpecs.at(1);
		if ((featureType_ == sequences) || (featureType_ == sequencelabels))
			caches_.back().nSequences = headerSpecs.at(2);
		else
			caches_.back().nSequences = 0;
	}

	/* if image or video bundle: create a cache specifier for each image/video */
	else if ((featureType_ == images) || (featureType_ == videos)) {
		//gets image specifications (width height and channels)
		width_ = headerSpecs.at(0);
		height_ = headerSpecs.at(1);
		channels_ = headerSpecs.at(2);
		require_gt(width_, 0);
		require_gt(height_, 0);
		if(channels_ != 1 && channels_ != 3)
			Core::Error::msg() << "FeatureCache: " << cacheFilename << " image channels should be either 1 or 3." << Core::Error::abort;
		featureDim_ = width_ * height_ * channels_;

		std::string line;
		while (cacheFile_->getline(line)) {
			caches_.push_back(CacheSpecifier());
			if (featureType_ == videos) {
				do {
					caches_.back().cacheFilename.push_back(line);
				} while (cacheFile_->getline(line) && (line.compare("#") != 0));
			}
			else { // type_ == image
				caches_.back().cacheFilename.push_back(line);
			}
			caches_.back().cacheSize = caches_.back().cacheFilename.size(); // 1 for images, nFrames for videos
			caches_.back().nSequences = (featureType_ == images ? 0 : 1);
		}
		cacheFile_->close();
	}

	// update global cache variables
	for (u32 i = 0; i < caches_.size(); i++) {
		cacheSize_ += caches_[i].cacheSize;
		nSequences_ += caches_[i].nSequences;
	}
}

const Math::Matrix<Float>& FeatureCache::next() {
	fillInputBuffer();
	return inputBuffer_;
}

void FeatureCache::logCacheInformation(const std::string& cacheFilename) {
	Core::Log::openTag("feature-cache.information", cacheFilename.c_str());
	switch (featureType_) {
	case vectors: Core::Log::os("feature type: vectors"); break;
	case sequences: Core::Log::os("feature type: sequences"); break;
	case images: Core::Log::os("feature type: images"); break;
	case videos: Core::Log::os("feature type: videos"); break;
	case labels: Core::Log::os("feature type: labels"); break;
	case sequencelabels: Core::Log::os("feature type: sequencelabels"); break;
	default: break;
	}
	Core::Log::os("total number of feature vectors: ") << cacheSize_;
	Core::Log::os("feature vector dimension: ") << featureDim_;
	if ((featureType_ == sequences) || (featureType_ == videos) || (featureType_ == sequencelabels))
		Core::Log::os("number of feature sequences: ") << nSequences_;
	Core::Log::closeTag();
}

FeatureCache::FeatureType FeatureCache::featureType(const std::string& cachefile) {
	Core::IOStream* stream;
	// open file as IOStream
	if (Core::Utils::isGz(cachefile))
		stream = new Core::CompressedStream(cachefile, std::ios::in);
	else if (Core::Utils::isBinary(cachefile))
		stream = new Core::BinaryStream(cachefile, std::ios::in);
	else
		stream = new Core::AsciiStream(cachefile, std::ios::in);
	FeatureCache::FeatureType type = getType(stream);
	stream->close();
	return type;
}
