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
