#ifndef ACTIONDETECTION_APPLICATION_HH_
#define ACTIONDETECTION_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"
#include "LengthModelling.hh"

namespace ActionDetection {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterString paramDetection_;
	enum Actions { none, linearSearch, slidingWindowSearch, estimatePoissonModel };

	void _rescore();
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* ACTIONDETECTION_APPLICATION_HH_ */
