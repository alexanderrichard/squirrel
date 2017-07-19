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

/*
 * Application.cc
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#include <iostream>
#include "Application.hh"
#include "Trainer.hh"
#include "Forwarder.hh"

using namespace Nn;

APPLICATION(Nn::Application)

const Core::ParameterEnum NeuralNetworkApplication::paramAction_("action", "none, training, forwarding", "none" , "");

const Core::ParameterInt NeuralNetworkApplication::paramBatchSize_("batch-size", 1, "");

void Application::main() {
	NeuralNetworkApplication app;
	app.run();
}

NeuralNetworkApplication::NeuralNetworkApplication() :
		batchSize_(Core::Configuration::config(paramBatchSize_))
{
	// batch size must be at least 1
	require_ge(batchSize_, 1);
}

void NeuralNetworkApplication::run() {
	switch ((Action)Core::Configuration::config(paramAction_)) {
	case training:
	{
		Trainer* trainer = Trainer::createTrainer();
		trainer->initialize();
		trainer->processAllEpochs(batchSize_);
		trainer->finalize();
	}
	break;
	case forwarding:
	{
		Forwarder forwarder;
		forwarder.initialize();
		forwarder.forward(batchSize_);
		forwarder.finalize();
	}
	break;
	case none:
	default:
		Core::Error::msg("No action given.") << Core::Error::abort;
		break;
	}
}
