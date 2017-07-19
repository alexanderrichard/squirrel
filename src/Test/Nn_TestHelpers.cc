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
 * Nn_TestHelpers.cc
 *
 *  Created on: Jun 3, 2014
 *      Author: richard
 */

#include "Nn_TestHelpers.hh"

Nn::NeuralNetwork* configureFeedForwardNetwork() {
	Core::Configuration::setParameter("neural-network.input-dimension", "2");
	Core::Configuration::setParameter("neural-network.connections", "conn-0-1,conn-1-2");
	Core::Configuration::setParameter("neural-network.conn-0-1.from", "network-input");
	Core::Configuration::setParameter("neural-network.conn-0-1.to", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-2.from", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-2.to", "layer-2");
	Core::Configuration::setParameter("neural-network.layer-1.type", "sigmoid");
	Core::Configuration::setParameter("neural-network.layer-1.number-of-units", "2");
	Core::Configuration::setParameter("neural-network.layer-2.type", "softmax");
	Core::Configuration::setParameter("neural-network.layer-2.number-of-units", "3");

	Nn::NeuralNetwork* network = new Nn::NeuralNetwork();

	return network;
}

void modifyFeedForwardNetwork(Nn::NeuralNetwork* network) {
	// reset computing state to set weights manually
	network->finishComputation();
	// set weights and bias for layer-1
	network->layer(0).weights(0,0).at(0,0) = 1;
	network->layer(0).weights(0,0).at(0,1) = 2;
	network->layer(0).weights(0,0).at(1,0) = -2;
	network->layer(0).weights(0,0).at(1,1) = -3;
	network->layer(0).bias(0).at(0) = 2;
	network->layer(0).bias(0).at(1) = 7;
	// set weights and bias for layer-2
	network->layer(1).weights(0,0).at(0,0) = -0.5;
	network->layer(1).weights(0,0).at(0,1) = 1.5;
	network->layer(1).weights(0,0).at(0,2) = -1;
	network->layer(1).weights(0,0).at(1,0) = 0.5;
	network->layer(1).weights(0,0).at(1,1) = -1;
	network->layer(1).weights(0,0).at(1,2) = 2;
	network->layer(1).bias(0).at(0) = 0;
	network->layer(1).bias(0).at(1) = 1;
	network->layer(1).bias(0).at(2) = -1;
}

Nn::NeuralNetwork* configureRecurrentNetwork() {
	Core::Configuration::setParameter("neural-network.input-dimension", "2");
	Core::Configuration::setParameter("neural-network.connections", "conn-0-1,conn-1-1,conn-1-2,conn-2-2");
	Core::Configuration::setParameter("neural-network.conn-0-1.from", "network-input");
	Core::Configuration::setParameter("neural-network.conn-0-1.to", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-1.from", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-1.to", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-2.from", "layer-1");
	Core::Configuration::setParameter("neural-network.conn-1-2.to", "layer-2");
	Core::Configuration::setParameter("neural-network.conn-2-2.from", "layer-2");
	Core::Configuration::setParameter("neural-network.conn-2-2.to", "layer-2");
	Core::Configuration::setParameter("neural-network.layer-1.type", "sigmoid");
	Core::Configuration::setParameter("neural-network.layer-1.number-of-units", "2");
	Core::Configuration::setParameter("neural-network.layer-2.type", "softmax");
	Core::Configuration::setParameter("neural-network.layer-2.number-of-units", "3");

	Nn::NeuralNetwork* network = new Nn::NeuralNetwork;

	return network;
}

void modifyRecurrentNetwork(Nn::NeuralNetwork* network) {
	network->setMaximalMemory(2);
	network->finishComputation();
	// set weights and bias for layer-1/connection-0-1
	network->layer(0).weights(0,0).at(0,0) = 1;
	network->layer(0).weights(0,0).at(0,1) = 2;
	network->layer(0).weights(0,0).at(1,0) = -2;
	network->layer(0).weights(0,0).at(1,1) = -3;
	network->layer(0).bias(0).at(0) = 2;
	network->layer(0).bias(0).at(1) = 7;
	// set weights for connection-1-1
	network->layer(0).weights(1,0).at(0,0) = -0.5;
	network->layer(0).weights(1,0).at(0,1) = 1.5;
	network->layer(0).weights(1,0).at(1,0) = 0.0;
	network->layer(0).weights(1,0).at(1,1) = -2.5;
	// set weights and bias for layer-2/connection-1-2
	network->layer(1).weights(0,0).at(0,0) = -0.5;
	network->layer(1).weights(0,0).at(0,1) = 1.5;
	network->layer(1).weights(0,0).at(0,2) = -1;
	network->layer(1).weights(0,0).at(1,0) = 0.5;
	network->layer(1).weights(0,0).at(1,1) = -1;
	network->layer(1).weights(0,0).at(1,2) = 2;
	network->layer(1).bias(0).at(0) = 0;
	network->layer(1).bias(0).at(1) = 1;
	network->layer(1).bias(0).at(2) = -1;
	// set weights for connection-2-2
	network->layer(1).weights(1,0).at(0,0) = 1.5;
	network->layer(1).weights(1,0).at(0,1) = 0.0;
	network->layer(1).weights(1,0).at(0,2) = -0.5;
	network->layer(1).weights(1,0).at(1,0) = 1.0;
	network->layer(1).weights(1,0).at(1,1) = -1.0;
	network->layer(1).weights(1,0).at(1,2) = -1.5;
	network->layer(1).weights(1,0).at(2,0) = -0.5;
	network->layer(1).weights(1,0).at(2,1) = 0.5;
	network->layer(1).weights(1,0).at(2,2) = 2.0;
}
