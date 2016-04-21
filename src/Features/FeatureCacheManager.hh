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

#ifndef FEATURES_FEATURECACHEMANAGER_HH_
#define FEATURES_FEATURECACHEMANAGER_HH_

#include "Core/CommonHeaders.hh"
#include "FeatureReader.hh"
#include "FeatureWriter.hh"
#include "LabeledFeatureReader.hh"
#include "LabelWriter.hh"

namespace Features {

/*
 * feature cache manager for subsampling or printing a feature cache
 */
class FeatureCacheManager
{
protected:
	bool isSingleCache_;
public:
	FeatureCacheManager();
	virtual ~FeatureCacheManager() {}
	virtual void initialize() {}
	/*
	 * apply the functionality of the instance of the feature cache manager
	 */
	virtual void work() = 0;
};

class FeatureCachePrinter : public FeatureCacheManager
{
private:
	void printSingleCache();
	void printSingleLabelCache();
	void printSequenceCache();
	void printSequenceLabelCache();
public:
	FeatureCachePrinter(bool isSingleCache);
	virtual ~FeatureCachePrinter() {}
	virtual void work();
};

class Subsampler : public FeatureCacheManager
{
private:
	static const Core::ParameterInt paramMaxNumberOfSamples_;
	static const Core::ParameterEnum paramSamplingMode_;
	enum SamplingMode { uniform, random };
	typedef FeatureCacheManager Precursor;
	u32 maxSamples_;
	SamplingMode samplingMode_;
	std::vector<u32> selection_;
private:
	void randomSelection(u32 nFrames);
	void uniformSelection(u32 nFrames);
	void subsampleSingleCache();
	void subsampleSequenceCache();
public:
	Subsampler(bool isSingleCache);
	virtual ~Subsampler() {}
	virtual void initialize();
	virtual void work();
};

}

#endif /* FEATURES_FEATURECACHEMANAGER_HH_ */
