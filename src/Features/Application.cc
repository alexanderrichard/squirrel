#include "Application.hh"
#include "FeatureReader.hh"
#include "FeatureCacheManager.hh"
#include <iostream>

using namespace Features;

APPLICATION(Features::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, print-cache, cache-combination, subsampling",
		"none");

const Core::ParameterBool Application::paramTreatSequenceCacheAsSingleCache_("treat-sequence-cache-as-single-cache", false);

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case printCache:
	case subsampling:
		{
		FeatureCacheManager* mngr = createFeatureCacheManager();
		mngr->initialize();
		mngr->work();
		delete mngr;
		}
		break;
	case cacheCombination:
		{
		BaseCacheCombination* conc = createCacheCombiner();
		conc->initialize();
		conc->combine();
		conc->finalize();
		delete conc;
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}

FeatureCacheManager* Application::createFeatureCacheManager() {
	FeatureCacheManager* mngr = 0;
	u32 isSingleCache = false;
	if (Core::Configuration::config(paramTreatSequenceCacheAsSingleCache_)) {
		isSingleCache = true;
	}

	switch (Core::Configuration::config(paramAction_)) {
	case printCache:
		mngr = new FeatureCachePrinter(isSingleCache);
		Core::Log::os("Created FeatureCachePrinter.");
		break;
	case subsampling:
		mngr = new Subsampler(isSingleCache);
		Core::Log::os("Created Subsampler.");
		break;
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}

	require(mngr);
	return mngr;
}

BaseCacheCombination* Application::createCacheCombiner() {
	// create feature cache manager
	HeaderReader dummy("cache-combination.feature-reader-1");
	dummy.initialize();
	BaseCacheCombination* conc = 0;
	if ((dummy.getFeatureType() == FeatureCache::single) ||
			(Core::Configuration::config(paramTreatSequenceCacheAsSingleCache_))) {
		conc = new CacheCombination();
		Core::Log::os("Created CacheCombination.");
	}
	else if (dummy.getFeatureType() == FeatureCache::sequence) {
		conc = new SequenceCacheCombination();
		Core::Log::os("Created SequenceCacheCombination.");
	}
	require(conc);
	return conc;
}
