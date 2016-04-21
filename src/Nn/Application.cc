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
