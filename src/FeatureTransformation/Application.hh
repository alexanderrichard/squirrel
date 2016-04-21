#ifndef FEATURETRANSFORMATION_APPLICATION_HH_
#define FEATURETRANSFORMATION_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"

namespace FeatureTransformation {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	enum Actions { none, kernelTransformation, featureQuantization, temporalFeatureQuantization,
		meanAndVarianceEstimation, minMaxEstimation, sequenceIntegration, sequenceSegmentation, windowedSequenceSegmentation,
		framePadding, principalComponentAnalysis };
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace


#endif /* FEATURETRANSFORMATION_APPLICATION_HH_ */
