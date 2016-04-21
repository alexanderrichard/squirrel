#include <iostream>
#include "Application.hh"
#include "FeatureProcessor.hh"
#include <Features/FeatureWriter.hh>

using namespace Nn;

APPLICATION(Nn::Application)

const Core::ParameterEnum Application::paramAction_("action", "none, neural-network-training", "none");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case neuralNetworkTraining:
		invokeNeuralNetworkTraining();
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}

void Application::invokeNeuralNetworkTraining() {
	BaseFeatureProcessor* featureProcessor = BaseFeatureProcessor::createFeatureProcessor();

	featureProcessor->initialize();
	featureProcessor->processAllEpochs();
	featureProcessor->finalize();

	delete featureProcessor;
}
