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

#include "CacheCombination.hh"
#include <sstream>

using namespace Features;

/*
 * BaseCacheCombination
 */
const Core::ParameterInt BaseCacheCombination::paramNumberOfCaches_("number-of-caches", 2, "cache-combination");

const Core::ParameterFloatList BaseCacheCombination::paramCacheWeights_("cache-weights", "", "cache-combination");

const Core::ParameterEnum BaseCacheCombination::paramCombinationMethod_("combination-method",
		"concatenation, max-pooling, average-pooling, sum-pooling",
		"concatenation", "cache-combination");

BaseCacheCombination::BaseCacheCombination() :
	nCaches_(Core::Configuration::config(paramNumberOfCaches_)),
	cacheWeights_(Core::Configuration::config(paramCacheWeights_)),
	method_((CombinationMethod) Core::Configuration::config(paramCombinationMethod_))
{
	require((cacheWeights_.size() == nCaches_) || (cacheWeights_.size() == 0));
	if (cacheWeights_.size() == 0) {
		Core::Log::os("No cache weights specified. Weight caches uniformly.");
		cacheWeights_.resize(nCaches_, 1.0);
	}
}

void BaseCacheCombination::combine() {
	switch (method_) {
	case concatenate:
		Core::Log::os("Concatenate caches.");
		concat();
		break;
	case maxPooling:
		Core::Log::os("Combine caches using max-pooling.");
		pool();
		break;
	case averagePooling:
		Core::Log::os("Combine caches using average-pooling.");
		pool();
		break;
	case sumPooling:
		Core::Log::os("Combine caches using sum-pooling.");
		pool();
		break;
	default:
		; // this can not happen
	}
}

/*
 * CacheCombination
 */
CacheCombination::CacheCombination() :
	Precursor()
{}

CacheCombination::~CacheCombination() {
	for (u32 i = 0; i < featureReader_.size(); i++) {
		delete featureReader_.at(i);
	}
	featureReader_.clear();
}

void CacheCombination::initialize() {
	for (u32 i = 0; i < nCaches_; i++) {
		std::stringstream s;
		s << "cache-combination.feature-reader-" << i + 1;
		Features::FeatureReader *featureReader = new Features::FeatureReader(s.str().c_str());
		featureReader->initialize();
		featureReader_.push_back(featureReader);
		if (i > 0) {
			require_eq(featureReader_.at(i-1)->totalNumberOfFeatures(), featureReader_.at(i)->totalNumberOfFeatures());
			if ((method_ == averagePooling) || (method_ == maxPooling) || (method_ == sumPooling)) {
				require_eq(featureReader_.at(i-1)->featureDimension(), featureReader_.at(i)->featureDimension());
			}
		}
	}
}

void CacheCombination::finalize() {
	featureWriter_.finalize();
}

void CacheCombination::concat() {
	u32 dim = 0;
	for (u32 i = 0; i < featureReader_.size(); i++) {
		dim += featureReader_.at(i)->featureDimension();
	}
	Math::Vector<Float> f(dim);
	while (featureReader_.at(0)->hasFeatures()) {
		u32 offset = 0;
		for (u32 i = 0; i < featureReader_.size(); i++) {
			const Math::Vector<Float>& g = featureReader_.at(i)->next();
			for (u32 d = 0; d < featureReader_.at(i)->featureDimension(); d++) {
				f.at(d + offset) = g.at(d) * cacheWeights_.at(i);
			}
			offset += featureReader_.at(i)->featureDimension();
		}
		featureWriter_.write(f);
	}
}

void CacheCombination::pool() {
	u32 dim = featureReader_.at(0)->featureDimension();
	while (featureReader_.at(0)->hasFeatures()) {
		Math::Vector<Float> f(dim);
		f.setToZero();
		for (u32 i = 0; i < featureReader_.size(); i++) {
			const Math::Vector<Float>& g = featureReader_.at(i)->next();
			for (u32 d = 0; d < dim; d++) {
				if (method_ == maxPooling) {
					f.at(d) = std::max(f.at(d), g.at(d) * cacheWeights_.at(i));
				}
				else { // if method_ == averagePooling or method == sumPooling
					f.at(d) += g.at(d) * cacheWeights_.at(i);
				}
			}
		}
		if (method_ == averagePooling)
			f.scale(1.0 / featureReader_.size());
		featureWriter_.write(f);
	}
}

/*
 * SequenceCacheCombination
 */
SequenceCacheCombination::SequenceCacheCombination() :
		Precursor()
{}

SequenceCacheCombination::~SequenceCacheCombination() {
	for (u32 i = 0; i < featureReader_.size(); i++) {
		delete featureReader_.at(i);
	}
	featureReader_.clear();
}

void SequenceCacheCombination::initialize() {
	for (u32 i = 0; i < nCaches_; i++) {
		std::stringstream s;
		s << "cache-combination.feature-reader-" << i + 1;
		Features::SequenceFeatureReader *featureReader = new Features::SequenceFeatureReader(s.str().c_str());
		featureReader->initialize();
		featureReader_.push_back(featureReader);
		if (i > 0) {
			require_eq(featureReader_.at(i-1)->totalNumberOfSequences(), featureReader_.at(i)->totalNumberOfSequences());
			if ((method_ == averagePooling) || (method_ == maxPooling) || (method_ == sumPooling)) {
				require_eq(featureReader_.at(i-1)->featureDimension(), featureReader_.at(i)->featureDimension());
			}
		}
	}
}

void SequenceCacheCombination::finalize() {
	featureWriter_.finalize();
}

void SequenceCacheCombination::concat() {
	u32 dim = 0;
	for (u32 i = 0; i < featureReader_.size(); i++) {
		dim += featureReader_.at(i)->featureDimension();
	}
	while (featureReader_.at(0)->hasSequences()) {
		u32 offset = 0;
		Math::Matrix<Float> f;
		for (u32 i = 0; i < featureReader_.size(); i++) {
			const Math::Matrix<Float>& g = featureReader_.at(i)->next();
			if (i == 0) {
				f.resize(dim, g.nColumns());
			}
			else {
				require_eq(f.nColumns(), g.nColumns());
			}
			for (u32 col = 0; col < f.nColumns(); col++) {
				for (u32 d = 0; d < featureReader_.at(i)->featureDimension(); d++) {
					f.at(d + offset, col) = g.at(d, col) * cacheWeights_.at(i);
				}
			}
			offset += featureReader_.at(i)->featureDimension();
		}
		featureWriter_.write(featureReader_.at(0)->currentTimestamps(), f);
	}
}

void SequenceCacheCombination::pool() {
	u32 dim = featureReader_.at(0)->featureDimension();
	while (featureReader_.at(0)->hasSequences()) {
		Math::Matrix<Float> f;
		for (u32 i = 0; i < featureReader_.size(); i++) {
			const Math::Matrix<Float>& g = featureReader_.at(i)->next();
			if (i == 0) {
				f.resize(dim, g.nColumns());
				f.setToZero();
			}
			else {
				require_eq(f.nColumns(), g.nColumns());
			}
			for (u32 col = 0; col < f.nColumns(); col++) {
				for (u32 d = 0; d < dim; d++) {
					if (method_ == maxPooling) {
						f.at(d, col) = std::max(f.at(d, col), g.at(d, col) * cacheWeights_.at(i));
					}
					else { // if method_ == averagePooling or method == sumPooling
						f.at(d, col) += g.at(d, col) * cacheWeights_.at(i);
					}
				}
			}
		}
		if (method_ == averagePooling) {
			f.scale(1.0 / featureReader_.size());
		}
		featureWriter_.write(featureReader_.at(0)->currentTimestamps(), f);
	}
}
