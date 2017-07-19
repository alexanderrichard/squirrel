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
 * Regularizer.cc
 *
 *  Created on: May 22, 2014
 *      Author: richard
 */

#include "Regularizer.hh"

using namespace Nn;

/*
 * Regularizer
 */
const Core::ParameterEnum Regularizer::paramRegularizerType_("regularizer", "none, l2-regularizer", "none", "");

const Core::ParameterFloat Regularizer::paramRegularizationConstant_("regularization-constant", 0.0, "");

Regularizer::Regularizer() :
		regularizationConstant_(Core::Configuration::config(paramRegularizationConstant_))
{}

Regularizer* Regularizer::createRegularizer() {
	Regularizer* regularizer = 0;
	switch ( (RegularizerType) Core::Configuration::config(paramRegularizerType_) ) {
	case l2Regularizer:
		Core::Log::os("Create l2-regularizer.");
		regularizer = new L2Regularizer();
		break;
	case none:
	default:
		regularizer = new Regularizer();
		break;
	}
	return regularizer;
}

/*
 * L2Regularizer
 */
L2Regularizer::L2Regularizer() :
		Precursor()
{}

void L2Regularizer::addToObjectiveFunction(NeuralNetwork& network, Statistics& statistics) {
	require(statistics.needsObjectiveFunction());
	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).useBias()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++)
				statistics.addToObjectiveFunction(network.layer(l).bias(port).sumOfSquares() * regularizationConstant_ / 2);
		}
	}
	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights()) {
			statistics.addToObjectiveFunction(network.connection(c).weights().sumOfSquares() * regularizationConstant_ / 2);
		}
	}
}

void L2Regularizer::addToGradient(NeuralNetwork& network, Statistics& statistics) {
	require(statistics.needsGradient());
	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).useBias() && network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++)
				statistics.biasGradient(network.layer(l).name(), port).add(network.layer(l).bias(port), regularizationConstant_);
		}
	}
	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			statistics.weightsGradient(network.connection(c).name()).add(network.connection(c).weights(), regularizationConstant_);
		}
	}
}
