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

#include "FeatureCacheManager.hh"
#include <iostream>
#include "Math/Vector.hh"
#include "Math/Matrix.hh"
#include "Math/Random.hh"
#include <vector>
#include <sstream>
#include <algorithm>
#include <math.h>

using namespace Features;

/*
 * FeatureCacheManager
 */
FeatureCacheManager::FeatureCacheManager() :
		isSingleCache_(true)
{}

/*
 * FeatureCachePrinter
 */
FeatureCachePrinter::FeatureCachePrinter(bool isSingleCache)
{
	isSingleCache_ = isSingleCache;
}

void FeatureCachePrinter::printSingleCache() {
	FeatureReader featureReader;
	featureReader.initialize();
	std::cout << std::endl;
	while (featureReader.hasFeatures()) {
		const Math::Vector<Float>& f = featureReader.next();
		for (u32 i = 0; i < featureReader.featureDimension(); i++) {
			std::cout << f.at(i) << " ";
		}
		std::cout << std::endl;
	}
}

void FeatureCachePrinter::printSingleLabelCache() {
	LabelReader labelReader;
	labelReader.initialize();
	std::cout << std::endl;
	while (labelReader.hasFeatures()) {
		std::cout << labelReader.next() << std::endl;
	}
}

void FeatureCachePrinter::printSequenceCache() {
	SequenceFeatureReader featureReader;
	featureReader.initialize();
	std::cout << std::endl;
	while (featureReader.hasSequences()) {
		const Math::Matrix<Float>& f = featureReader.next();
		// print sequence length
		std::cout << f.nColumns() << std::endl;
		// print sequence of feature vectors
		for (u32 i = 0; i < f.nColumns(); i++) {
			// print time stamp
			std::cout << "(t=" << featureReader.currentTimestamps().at(i) << ") ";
			for (u32 j = 0; j < f.nRows(); j++) {
				std::cout << f.at(j,i) << " ";
			}
			std::cout << std::endl;
		}
	}
}

void FeatureCachePrinter::printSequenceLabelCache() {
	SequenceLabelReader labelReader;
	labelReader.initialize();
	std::cout << std::endl;
	while (labelReader.hasSequences()) {
		std::vector<u32>& labels = labelReader.next();
		// print sequence length
		std::cout << labels.size() << std::endl;
		// print sequence of feature vectors
		for (u32 i = 0; i < labels.size(); i++) {
			// print time stamp
			std::cout << "(t=" << labelReader.currentTimestamps().at(i) << ") ";
			std::cout << labels.at(i) << std::endl;
		}
	}
}

void FeatureCachePrinter::work() {
	HeaderReader dummy;
	dummy.initialize();
	bool isLabelCache = (dummy.getCacheFormat() == FeatureCache::labelCache);
	if (dummy.getFeatureType() == FeatureCache::single) {
		isSingleCache_ = true;
	}
	if ((isSingleCache_) && (!isLabelCache))
		printSingleCache();
	else if ((isSingleCache_) && (isLabelCache))
		printSingleLabelCache();
	else if ((!isSingleCache_) && (!isLabelCache))
		printSequenceCache();
	else
		printSequenceLabelCache();
}

/*
 * Subsampler
 */
const Core::ParameterInt Subsampler::paramMaxNumberOfSamples_("max-number-of-samples", 1, "features.feature-cache-manager");

const Core::ParameterEnum Subsampler::paramSamplingMode_("sampling-mode", "uniform, random", "random", "features.feature-cache-manager");

Subsampler::Subsampler(bool isSingleCache) :
		Precursor(),
		maxSamples_(Core::Configuration::config(paramMaxNumberOfSamples_)),
		samplingMode_((SamplingMode) Core::Configuration::config(paramSamplingMode_))
{
	isSingleCache_ = isSingleCache;
	require_gt(maxSamples_, 0);
}

void Subsampler::initialize() {
	Precursor::initialize();
	Math::RandomNumberGenerator::initializeSRand();
}

void Subsampler::randomSelection(u32 nFrames) {
	require_ge(nFrames, maxSamples_);
	selection_.resize(nFrames);
	for (u32 i = 0; i < nFrames; i++)
		selection_.at(i) = i;
	std::random_shuffle(selection_.begin(), selection_.end());
	selection_.resize(maxSamples_);
	std::sort(selection_.begin(), selection_.end());
}

void Subsampler::uniformSelection(u32 nFrames) {
	require_ge(nFrames, maxSamples_);
	Float x = Float(nFrames) / maxSamples_;
	selection_.clear();
	selection_.resize(maxSamples_);
	for (u32 i = 0; i < maxSamples_; i++) {
		selection_.at(i) = round(x * (i+1)) - 1;
	}
}

void Subsampler::subsampleSingleCache() {
	FeatureReader featureReader;
	FeatureWriter featureWriter;
	featureReader.initialize();

	maxSamples_ = std::min(featureReader.totalNumberOfFeatures(), maxSamples_);
	if (samplingMode_ == random)
		randomSelection(featureReader.totalNumberOfFeatures());
	else // if samplingMode_ == uniform
		uniformSelection(featureReader.totalNumberOfFeatures());

	u32 j = 0;
	for (u32 i = 0; i < featureReader.totalNumberOfFeatures(); i++) {
		const Math::Vector<Float>& v = featureReader.next();
		if ((j < selection_.size()) && ( i == selection_.at(j))) {
			featureWriter.write(v);
			j++;
		}
	}
}

void Subsampler::subsampleSequenceCache() {
	SequenceFeatureReader featureReader;
	SequenceFeatureWriter featureWriter;
	featureReader.initialize();

	while (featureReader.hasSequences()) {
		const Math::Matrix<Float>& m = featureReader.next();
		// copy sequence if number of samples <= maxSamples_
		if (m.nColumns() <= maxSamples_) {
			featureWriter.write(featureReader.currentTimestamps(), m);
		}
		// else subsample matrix
		else {
			if (samplingMode_ == random)
				randomSelection(m.nColumns());
			else // if samplingMode_ == uniform
				uniformSelection(m.nColumns());
			Math::Matrix<Float> tmpSequence(m.nRows(), maxSamples_);
			std::vector<u32> tmpTimestamps(maxSamples_);
			for (u32 i = 0; i < maxSamples_; i++) {
				tmpSequence.copyBlockFromMatrix(m,
						0, selection_.at(i),
						0, i,
						m.nRows(), 1);
				tmpTimestamps.at(i) = featureReader.currentTimestamps().at(selection_.at(i));
			}
			featureWriter.write(tmpTimestamps, tmpSequence);
		}
	}
}

void Subsampler::work() {
	HeaderReader dummy;
	dummy.initialize();
	if (dummy.getFeatureType() == FeatureCache::single) {
		isSingleCache_ = true;
	}
	if (isSingleCache_)
		subsampleSingleCache();
	else
		subsampleSequenceCache();
}
