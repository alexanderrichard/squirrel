#ifndef CONVERTER_APPLICATION_HH_
#define CONVERTER_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"

namespace Converter {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterEnum paramExternalType_;
	static const Core::ParameterEnum paramInternalCacheConversions_;

	enum Action { none, internalCacheConversion, externalToCache, cacheToExternal, libSvmToLogLinear};
	enum ExternalType { asciiFeatureCache, asciiLabels, denseTrajectories, libSvm };
	enum InteralCacheConversions { singleLabelToSequenceLabel, matrixToSingleCache };

	void invokeInternalCacheConversion();
	void invokeExternalToCache();
	void invokeCacheToExternal();
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace


#endif /* CONVERTER_APPLICATION_HH_ */
