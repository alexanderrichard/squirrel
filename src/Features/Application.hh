#ifndef FEATURES_APPLICATION_HH_
#define FEATURES_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"
#include "FeatureCacheManager.hh"
#include "CacheCombination.hh"

namespace Features {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterBool paramTreatSequenceCacheAsSingleCache_;
	enum Actions { none, printCache, cacheCombination, subsampling};

	FeatureCacheManager* createFeatureCacheManager();
	BaseCacheCombination* createCacheCombiner();
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* APPLICATION_HH_ */
