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

#include "PortFusionLayer.hh"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "../Core/Types.hh"
#include "../Core/Utils.hh"
#include "../Math/CudaMatrix.hh"
#include "MatrixContainer.hh"
#include "NeuralNetwork.hh"

using namespace Nn;

/*
 * PortFusionLayer
 */
PortFusionLayer::PortFusionLayer(const char* name) :
		Precursor(name)
{}

void PortFusionLayer::initialize(u32 maxMemory) {
	Precursor::initialize(maxMemory);
	outputActivations_.setMaximalMemory(maxMemory);
	if (trainingMode_)
		outputErrorSignals_.setMaximalMemory(maxMemory);
}

Matrix& PortFusionLayer::activationsOut(u32 timeframe, u32 port) {
	require_eq(port, 0);
	require_lt(timeframe, nTimeframes());
	return outputActivations_.at(timeframe);
}

Matrix& PortFusionLayer::errorSignalOut(u32 timeframe, u32 port) {
	require(trainingMode_);
	require_eq(port, 0);
	require_lt(timeframe, nTimeframes());
	return outputErrorSignals_.at(timeframe);
}

void PortFusionLayer::addTimeframe(u32 minibatchSize, bool initWithZero) {
	Precursor::addTimeframe(minibatchSize, initWithZero);
	outputActivations_.addTimeframe(nUnits_, minibatchSize);
	if (initWithZero)
		outputActivations_.getLast().setToZero();
	if (trainingMode_) {
		outputErrorSignals_.addTimeframe(nUnits_, minibatchSize);
		if (initWithZero)
			outputErrorSignals_.getLast().setToZero();
	}
}

void PortFusionLayer::addEmptyTimeframe() {
	Precursor::addEmptyTimeframe();
	outputActivations_.addTimeframe(0, 0);
	if (trainingMode_)
		outputErrorSignals_.addTimeframe(0, 0);
}

void PortFusionLayer::setMaximalMemory(u32 maxMemory) {
	Precursor::setMaximalMemory(maxMemory);
	outputActivations_.setMaximalMemory(maxMemory);
	if (trainingMode_)
		outputErrorSignals_.setMaximalMemory(maxMemory);
}

void PortFusionLayer::resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns) {
	Precursor::resizeTimeframe(timeframe, nRows, nColumns);
	outputActivations_.at(timeframe).resize(nRows, nColumns);
	if (trainingMode_)
		outputErrorSignals_.at(timeframe).resize(nRows, nColumns);
}

void PortFusionLayer::reset() {
	Precursor::reset();
	outputActivations_.reset();
	if (trainingMode_)
		outputErrorSignals_.reset();
}

void PortFusionLayer::setActivationVisibility(u32 timeframe, u32 nVisibleColumns) {
	Precursor::setActivationVisibility(timeframe, nVisibleColumns);
	outputActivations_.at(timeframe).setVisibleColumns(nVisibleColumns);
}

void PortFusionLayer::setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns) {
	Precursor::setErrorSignalVisibility(timeframe, nVisibleColumns);
	outputErrorSignals_.at(timeframe).setVisibleColumns(nVisibleColumns);
}

void PortFusionLayer::initComputation(bool sync) {
	Precursor::initComputation(sync);
	outputActivations_.initComputation(sync);
	if (trainingMode_)
		outputErrorSignals_.initComputation(sync);
}

void PortFusionLayer::finishComputation(bool sync) {
	Precursor::finishComputation(sync);
	outputActivations_.finishComputation(sync);
	if (trainingMode_)
		outputErrorSignals_.finishComputation(sync);
}

/*
 * MaxoutLayer
 */
MaxoutLayer::MaxoutLayer(const char* name) :
		Precursor(name)
{}

void MaxoutLayer::forward() {
	Precursor::forward();
	u32 t = nTimeframes() - 1;
	activationsOut(t, 0).copy(activationsIn(t, 0));
	for (u32 port = 1; port < nInputPorts(); port++) {
		activationsOut(t, 0).max(activationsOut(t, 0), activationsIn(t, port));
	}
}

void MaxoutLayer::backpropagate(u32 timeframe) {
	Precursor::backpropagate(timeframe);
	u32 t = timeframe;
	for (u32 port = 0; port < nInputPorts(); port++) {
		errorSignalIn(t, port).copy(errorSignalOut(t, 0));
		// Note: this is not the correct derivative if two input ports have the same value. However, this occurs only rarely.
		errorSignalIn(t, port).elementwiseMultiplicationWithKroneckerDelta(activationsOut(t, 0), activationsIn(t, port));
	}
}

/*
 * GatedRecurrentUnitLayer
 */
const Core::ParameterString GatedRecurrentUnitLayer::paramOldGruFilename_("load-gru-weights-from", "", "neural-network.layer");

const Core::ParameterString GatedRecurrentUnitLayer::paramNewGruFilename_("write-gru-weights-to", "", "neural-network.layer");

GatedRecurrentUnitLayer::GatedRecurrentUnitLayer(const char* name) :
		Precursor(name)
{
	nPorts_ = 4; // four input ports, only three are "visible" via nInputPorts()
}

void GatedRecurrentUnitLayer::blockPorts() {
	blockedInputPorts_.resize(nInputPorts(), false);
	blockedInputPorts_.back() = true;
}

void GatedRecurrentUnitLayer::unblockPorts() {
	blockedInputPorts_.resize(nInputPorts(), false);
	blockedInputPorts_.back() = false;
}

bool GatedRecurrentUnitLayer::isInputPortBlocked(u32 port) {
	if ((blockedInputPorts_.size() > port) && (blockedInputPorts_.at(port) == true))
		return true;
	else
		return false;
}

// this function is called in Nn::NeuralNetwork directly after the layer constructor
void GatedRecurrentUnitLayer::addInternalConnections(NeuralNetwork& network) {
	// port 0, 1, 2 can be accessed by previous layer to provide recent input
	// port 0, 1, 3 are used for recurrent connections (update gate, reset gate, recurrent part of candidate activation)
	std::string connectionName = name();
	connectionName.append(".gru-connections");
	// recurrent connection for update gate (port 0)
	network.addConnection(connectionName, (BaseLayer*)this, (BaseLayer*)this, 0, 0, true);
	// recurrent connection for reset gate (port 1)
	network.addConnection(connectionName, (BaseLayer*)this, (BaseLayer*)this, 0, 1, true);
	// recurrent connection for old information flow (port 3)
	network.addConnection(connectionName, (BaseLayer*)this, (BaseLayer*)this, 0, 3, true);
	// set parameter file for the three connections
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 0)) )->setOldWeightsFile(Core::Configuration::config(paramOldGruFilename_, prefix_));
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 0)) )->setNewWeightsFile(Core::Configuration::config(paramNewGruFilename_, prefix_));
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 1)) )->setOldWeightsFile(Core::Configuration::config(paramOldGruFilename_, prefix_));
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 1)) )->setNewWeightsFile(Core::Configuration::config(paramNewGruFilename_, prefix_));
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 3)) )->setOldWeightsFile(Core::Configuration::config(paramOldGruFilename_, prefix_));
	dynamic_cast<WeightConnection*>( &(incomingConnection(0, 3)) )->setNewWeightsFile(Core::Configuration::config(paramNewGruFilename_, prefix_));
}

void GatedRecurrentUnitLayer::forward() {
	Precursor::forward();
	u32 t = nTimeframes() - 1;
	// sigmoid on update gate (port 0)
	activationsIn(t, 0).sigmoid();
	// sigmoid on reset gate (port 1)
	activationsIn(t, 1).sigmoid();
	// transform input of port 2 to candidate activation
	activationsOut(t, 0).copy(activationsIn(t, 1));
	activationsOut(t, 0).elementwiseMultiplication(activationsIn(t, 3));
	activationsIn(t, 2).add(activationsOut(t, 0));
	activationsIn(t, 2).tanh();
	// compute output activation
	activationsOut(t, 0).copy(activationsIn(t, 2));
	if (t > 0)
		activationsOut(t, 0).add(activationsOut(t-1, 0), (Float)-1.0);
	activationsOut(t, 0).elementwiseMultiplication(activationsIn(t, 0));
	if (t > 0)
		activationsOut(t, 0).add(activationsOut(t-1, 0));
}

void GatedRecurrentUnitLayer::backpropagate(u32 timeframe) {
	Precursor::backpropagate(timeframe);
	u32 t = timeframe;
	// error signal for update gate (port 0)
	errorSignalIn(t, 0).copy(activationsIn(t, 2));
	if (t > 0) {
		errorSignalIn(t, 0).safeResize(errorSignalIn(t, 0).nRows(), activationsOut(t-1, 0).nColumns());
		errorSignalIn(t, 0).add(activationsOut(t-1, 0), (Float)-1.0);
		errorSignalIn(t, 0).safeResize(errorSignalIn(t, 0).nRows(), activationsOut(t, 0).nColumns());
	}
	errorSignalIn(t, 0).elementwiseMultiplicationWithSigmoidDerivative(activationsIn(t, 0));
	errorSignalIn(t, 0).elementwiseMultiplication(errorSignalOut(t, 0));
	// error signal for port 2
	errorSignalIn(t, 2).copy(errorSignalOut(t, 0));
	errorSignalIn(t, 2).elementwiseMultiplicationWithTanhDerivative(activationsIn(t, 2));
	errorSignalIn(t, 2).elementwiseMultiplication(activationsIn(t, 0));
	// error signal for reset gate (port 1)
	errorSignalIn(t, 1).copy(errorSignalIn(t, 2));
	errorSignalIn(t, 1).elementwiseMultiplicationWithSigmoidDerivative(activationsIn(t, 1));
	errorSignalIn(t, 1).elementwiseMultiplication(activationsIn(t, 3));
	// error signal for recurrent connection at port 3
	errorSignalIn(t, 3).copy(errorSignalIn(t, 2));
	errorSignalIn(t, 3).elementwiseMultiplication(activationsIn(t, 1));
}
