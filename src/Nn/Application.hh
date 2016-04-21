#ifndef NN_APPLICATION_HH_
#define NN_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"

namespace Nn {

class Application: public Core::Application
{
private:
	static const Core::ParameterEnum paramAction_;
	enum Actions { none, neuralNetworkTraining };
private:
	void invokeNeuralNetworkTraining();
public:
	virtual ~Application() {}
	virtual void main();
};

} // namespace

#endif /* NN_APPLICATION_HH_ */
