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

#include "NeuralNetwork.hh"
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <map>

using namespace Nn;

const Core::ParameterStringList NeuralNetwork::paramConnections_("connections", "", "neural-network");

const Core::ParameterInt NeuralNetwork::paramInputDimension_("input-dimension", 0, "neural-network");

NeuralNetwork::NeuralNetwork() :
		inputDimension_(Core::Configuration::config(paramInputDimension_)),
		nLayer_(0),
		isRecurrent_(false),
		isInitialized_(false),
		isComputing_(false)
{}

NeuralNetwork::~NeuralNetwork() {
	for (u32 i = 0; i < layer_.size(); i++) {
		delete layer_.at(i);
	}
	for (u32 i = 0; i < connections_.size(); i++) {
		delete connections_.at(i);
	}
}

bool NeuralNetwork::layerExists(std::string& layerName) {
	if ( (layerName.compare("network-input") == 0) ||
			(layerNameToIndex_.find(layerName) != layerNameToIndex_.end()) )
		return true;
	else
		return false;
}

void NeuralNetwork::addLayer(std::string& layerName) {
	if (layerName.compare("network-input") != 0) {
		layer_.push_back(Layer::createLayer(layerName.c_str()));
		// add internal connections, e.g. for complex recurrent units like LSTMs or GRUs
		layer_.back()->addInternalConnections(*this);
		layerNameToIndex_[layerName] = nLayer_;
		nLayer_++;
		if (layer_.back()->isRecurrent())
			isRecurrent_ = true;
	}
}

void NeuralNetwork::addConnection(Connection* connection, bool needsWeightsFileSuffix) {
	connections_.push_back(connection);

	if (connection->from().name().compare("network-input") != 0) { // input layer is no node in the graph (just an incoming arc)
		Layer* source = dynamic_cast<Layer*>(&(connection->from()));
		require(!source->isOutputPortBlocked(connection->sourcePort()));
		source->addOutgoingConnection(connection, connection->sourcePort());
	}
	Layer* dest = dynamic_cast<Layer*>(&(connection->to()));
	require(!dest->isInputPortBlocked(connection->destinationPort()));
	dest->addIncomingConnection(connection, connection->destinationPort());

	if (needsWeightsFileSuffix)
		connection->setWeightsFileSuffix();

	// check if the added connection is a recurrent connection
	if (connection->isRecurrent())
		isRecurrent_ = true;
}

void NeuralNetwork::addConnection(std::string& connectionName, BaseLayer* sourceLayer, BaseLayer* destLayer,
		u32 sourcePort, u32 destPort, bool needsWeightsFileSuffix) {
	Connection* newConnection = Connection::createConnection(connectionName.c_str(), sourceLayer, destLayer, sourcePort, destPort);
	addConnection(newConnection, needsWeightsFileSuffix);
}

void NeuralNetwork::buildNetwork() {
	std::vector<std::string> connections = Core::Configuration::config(paramConnections_);

	for (std::vector<std::string>::iterator it = connections.begin(); it != connections.end(); ++it) {
		/* create and configure source and destination layer if they do not yet exist */
		// get source and destination of connection
		std::string connectionPath("neural-network.");
		connectionPath.append(*it);
		std::string source = Core::Configuration::config(Core::ParameterString("from", "", connectionPath.c_str()));
		std::string dest = Core::Configuration::config(Core::ParameterString("to", "", connectionPath.c_str()));
		require(!source.empty());
		require(!dest.empty());
		// source and dest may be comprised of layer name and port
		std::vector<std::string> v;
		Core::Utils::tokenizeString(v, source, ":");
		source = v[0];
		s32 sourcePort = (v.size() > 1 ? atoi(v[1].c_str()) : -1);
		Core::Utils::tokenizeString(v, dest, ":");
		dest = v[0];
		s32 destPort = (v.size() > 1 ? atoi(v[1].c_str()) : -1);
		// create the corresponding layer
		if (!layerExists(source))
			addLayer(source);
		if(!layerExists(dest))
			addLayer(dest);
		// link connection with source and destination layer
		BaseLayer* sourceLayer = (source.compare("network-input") == 0) ?
				(BaseLayer*)&inputLayer_ : (BaseLayer*)layer_.at(layerNameToIndex_[source]);
		if (dest.compare("network-input") == 0) {
			std::cerr << "Error: network-input must not have incoming connections. Abort." << std::endl;
			exit(1);
		}
		BaseLayer* destLayer = layer_.at(layerNameToIndex_[dest]);

		// if the layer has only one input/output port, this port is the only candidate to be used
		if ((sourcePort == -1) && (sourceLayer->nOutputPorts() == 1))
			sourcePort = 0;
		if ((destPort == -1) && (destLayer->nInputPorts() == 1))
			destPort = 0;

		/* add connection */
		sourceLayer->blockPorts();
		destLayer->blockPorts();
		// case 1 (multiple source and dest ports):
		// if no port given for source and dest establish parallel connections (port i of source to port i of dest)
		if ((sourcePort == -1) && (destPort == -1)) {
			require_eq(sourceLayer->nOutputPorts(), destLayer->nInputPorts());
			for (u32 i = 0; i < sourceLayer->nOutputPorts(); i++) {
				if ((!sourceLayer->isOutputPortBlocked(i)) && (!destLayer->isInputPortBlocked(i)))
					addConnection(*it, sourceLayer, destLayer, i, i,
							( ((sourceLayer->nOutputPorts() == 1) && (destLayer->nInputPorts() == 1)) ? false : true));
			}
		}
		// case 2 (multiple source ports):
		// if no source port given but a destination port connect all ports of source to the given destination port (fan-in)
		else if (sourcePort == -1) {
			for (u32 i = 0; i < sourceLayer->nOutputPorts(); i++) {
				if ((!sourceLayer->isOutputPortBlocked(i)) && (!destLayer->isInputPortBlocked((u32)destPort)))
					addConnection(*it, sourceLayer, destLayer, i, (u32)destPort,
							(sourceLayer->nOutputPorts() == 1 ? false : true));
			}
		}
		// case 3 (multiple dest ports):
		// if no destination port given but a source port connect the given source port to all destination ports (fan-out)
		else if (destPort == -1) {
			for (u32 i = 0; i < destLayer->nInputPorts(); i++) {
				if ((!sourceLayer->isOutputPortBlocked((u32)sourcePort)) && (!destLayer->isInputPortBlocked(i)))
					addConnection(*it, sourceLayer, destLayer, (u32)sourcePort, i,
							(destLayer->nInputPorts() == 1 ? false : true));
			}
		}
		// case 4: connection from sourcePort to destPort
		else {
			if ((!sourceLayer->isOutputPortBlocked((u32)sourcePort)) && (!destLayer->isInputPortBlocked((u32)destPort)))
				addConnection(*it, sourceLayer, destLayer, (u32)sourcePort, (u32) destPort, false);
		}
		sourceLayer->unblockPorts();
		destLayer->unblockPorts();
	}
}

bool NeuralNetwork::containsLayer(BaseLayer* layer, std::vector<BaseLayer*> v) {
	for (std::vector<BaseLayer*>::iterator it = v.begin(); it != v.end(); ++it) {
		if (*it == layer)
			return true;
	}
	return false;
}

void NeuralNetwork::sortTopologically() {
	std::vector<BaseLayer*> sortedLayers;
	// at most layer_.size() many iterations required until topological order is established
	for (u32 dummy = 0; dummy < layer_.size(); dummy++) {
		// for each layer ...
		for (u32 l = 0; l < layer_.size(); l++) {
			// ... that is not yet sorted ...
			if (!containsLayer(layer_.at(l), sortedLayers)) {
				bool isNextInTopologicalOrder = true;
				// ... check if it is next in topological order (has only already sorted predecessors)
				for (u32 port = 0; port < layer_.at(l)->nInputPorts(); port++) {
					for (u32 i = 0; i < layer_.at(l)->nIncomingConnections(port); i++) {
						// if predecessor is not the input, not recurrent, and not yet sorted ...
						if ( (layer_.at(l)->incomingConnection(i,port).from().name().compare("network-input") != 0) &&
								(!layer_.at(l)->incomingConnection(i,port).isRecurrent()) &&
								(!containsLayer(&(layer_.at(l)->incomingConnection(i,port).from()), sortedLayers)) ) {
							// ... this layer cannot be the next in topological order
							isNextInTopologicalOrder = false;
						}
					}
				}
				// store the layer at the correct position and update mapping layerNameToIndex_
				if (isNextInTopologicalOrder) {
					sortedLayers.push_back(layer_.at(l));
					layerNameToIndex_.at(layer_.at(l)->name()) = sortedLayers.size() - 1;
				}
			}
		}
	}
	// check if topological order could be established
	if (sortedLayers.size() != layer_.size()) {
		std::cerr << "Neural network cannot be sorted topologically. "
				<< "Make sure your network is an acyclic directed graph (except for the recurrent connections)."
				<< std::endl;
		exit(1);
	}
	// store layers in topological order ...
	layer_.clear();
	for (u32 l = 0; l < sortedLayers.size(); l++) {
		layer_.push_back(dynamic_cast<Layer*>(sortedLayers.at(l)));
	}
}

void NeuralNetwork::logTopology() {
	Core::Log::openTag("neural-network.topology");
	for (u32 i = 0; i < layer_.size(); i++) {
		Core::Log::openTag(layer_[i]->name().c_str());
		for (u32 port = 0; port < layer_[i]->nInputPorts(); port++) {
			std::stringstream logMsg;
			logMsg << "incoming connections to input port " << port << ":";
			for (u32 j = 0; j < layer_[i]->nIncomingConnections(port); j++) {
				logMsg << " " << layer_[i]->incomingConnection(j, port).from().name() << ":port-"
						<< layer_[i]->incomingConnection(j, port).sourcePort();
			}
			Core::Log::os() << logMsg.str();
		}
		for (u32 port = 0; port < layer_[i]->nOutputPorts(); port++) {
			std::stringstream logMsg;
			logMsg << "outgoing connections from output port " << port << ":";
			for (u32 j = 0; j < layer_[i]->nOutgoingConnections(port); j++) {
				logMsg << " " << layer_[i]->outgoingConnection(j, port).to().name() << ":port-"
						<< layer_[i]->outgoingConnection(j, port).destinationPort();
			}
			Core::Log::os() << logMsg.str();
		}
		Core::Log::closeTag();
	}
	Core::Log::closeTag();
}

void NeuralNetwork::initialize(u32 maxMemory) {
	Core::Log::openTag("neural-network.initialize");
	// check if input dimension is given
	if (inputDimension_ == 0) {
		std::cerr << "Error: 0 is an invalid value for the parameter neural-network.input-dimension. Abort." << std::endl;
		exit(1);
	}
	buildNetwork();
	sortTopologically();
	// output layer has exactly one output port
	if (layer(nLayer_ - 1).nOutputPorts() != 1) {
		std::cerr << "Error: Output layer " << layer(nLayer_ - 1).name() << " must not have more than one output port. Abort." << std::endl;
		exit(1);
	}
	logTopology();
	// first initialize all layers...
	inputLayer_.initialize(inputDimension_);
	for (u32 l = 0; l < nLayer_; l++) {
		layer_.at(l)->initialize(maxMemory);
	}
	// ... then initialize the connections
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->initialize();
	}
	Core::Log::closeTag();
	isInitialized_ = true;
}

u32 NeuralNetwork::lastRecurrentLayerIndex() const {
	u32 index = 0;
	for (u32 l = 0; l < nLayer_; l++) {
		index = (layer(l).isRecurrent() ? l : index);
	}
	return index;
}

void NeuralNetwork::setMaximalMemory(u32 maxMemory) {
	require(maxMemory > 0);
	for (u32 l = 0; l < layer_.size(); l++) {
		layer_.at(l)->setMaximalMemory(maxMemory);
	}
	if (isRecurrent() && (maxMemory < 2)) {
		std::cerr << "WARNING: maxMemory should be at least 2 for recurrent neural networks." << std::endl;
	}
}

Layer& NeuralNetwork::layer(u32 layerIndex) const {
	require(layerIndex < nLayer_);
	return *(layer_.at(layerIndex));
}

Connection& NeuralNetwork::connection(u32 connectionIndex) const {
	require(isInitialized_);
	require(connectionIndex < connections_.size());
	return *(connections_.at(connectionIndex));
}

void NeuralNetwork::forward(Matrix& batch, u32 maxLayerIndex) {
	require(isInitialized_);
	maxLayerIndex = std::min(nLayer_ - 1, maxLayerIndex);
	// reset all activations
	reset();
	inputLayer_.addInput(batch);
	initComputation();

	// forward all layers
	for (u32 l = 0; l <= maxLayerIndex; l++) {
		// allocate storage for one time frame in the activations container...
		layer(l).addTimeframe(batch.nColumns());
		// ... and forward the input
		layer(l).forward();
		// apply dropout if desired
		if (layer(l).useDropout())
			layer(l).dropout();
		// finalize the forwarding for this layer (only implemented for special layer types)
		layer(l).finalizeForwarding();
	}
}

void NeuralNetwork::forwardTimeframe(MatrixContainer& batchedSequence, u32 t, bool greedyForwarding) {
	// number of active sequences at timeframe t+1
	u32 size = (t == batchedSequence.nTimeframes() - 1 ?
			batchedSequence.getLast().nColumns() : batchedSequence.at(t+1).nColumns());
	// add the input
	inputLayer_.addInput(batchedSequence.at(t));
	initComputation();

	u32 nLayer = nLayer_;
	if (greedyForwarding && (t < batchedSequence.nTimeframes() - 1)) {
		// forward layers up to last recurrent layer if final timeframe not yet reached, else forward all layers
		nLayer = lastRecurrentLayerIndex() + 1;
	}

	for (u32 l = 0; l < nLayer; l++) {
		// allocate storage for one time frame in the activations container...
		layer(l).addTimeframe(size,
				(layer(l).isRecurrent() ? true : false)); // init with zero if recurrent (necessary due visibility issues)
		// ... hide the elements that are currently not needed ...
		layer(l).setActivationVisibility(t, batchedSequence.at(t).nColumns());
		// ... unhide the elements from the previous time frame that may be needed due to recurrency...
		if ((t > 0) && layer(l).isRecurrent())
			layer(l).setActivationVisibility(t-1, batchedSequence.at(t).nColumns());
		// ... and forward the input
		layer(l).forward();
		// apply dropout if desired
		if (layer(l).useDropout())
			layer(l).dropout();
		// hide the elements that have been unhidden
		if ((t > 0) && layer(l).isRecurrent())
			layer(l).setActivationVisibility(t-1, batchedSequence.at(t-1).nColumns());
		// finalize the forwarding if the last time frame is reached
		if (t == batchedSequence.nTimeframes() - 1)
			layer(l).finalizeForwarding();
	}
	// add empty timeframes to every layer that has not been forwarded
	// empty timeframes do not allocate memory but just keep the timeframe numbering consistent
	for (u32 l = nLayer; l < nLayer_; l++) {
		layer(l).addEmptyTimeframe();
	}
}

void NeuralNetwork::forwardSequence(MatrixContainer& batchedSequence) {
	require(isInitialized_);
	// reset all activations
	reset();
	// forward all time frames
	for (u32 t = 0; t < batchedSequence.nTimeframes(); t++) {
		forwardTimeframe(batchedSequence, t);
	}
}

void NeuralNetwork::reset() {
	// reset all activations (and error signals if network is in training mode)
	for (u32 l = 0; l < layer_.size(); l++) {
		layer_.at(l)->reset();
	}
	// reset the input
	inputLayer_.reset();
}

void NeuralNetwork::setTrainingMode(bool trainingMode, u32 firstTrainableLayerIndex) {
	for (u32 l = firstTrainableLayerIndex; l < layer_.size(); l++) {
		layer_.at(l)->setTrainingMode(trainingMode);
	}
}

void NeuralNetwork::saveNeuralNetworkParameters() {
	std::string suffix = "";
	saveNeuralNetworkParameters(suffix);
}

void NeuralNetwork::saveNeuralNetworkParameters(const std::string& suffix) {
	require(isInitialized_);
	for (u32 l = 0; l < layer_.size(); l++) {
		if (layer_.at(l)->useBias())
			layer_.at(l)->saveBias(suffix);
	}
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->saveWeights(suffix);
	}
}

void NeuralNetwork::initComputation(bool sync) {
	require(isInitialized_);
	inputLayer_.initComputation(sync);
	for (u32 l = 0; l < nLayer_; l++) {
		layer_.at(l)->initComputation(sync);
	}
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->initComputation(sync);
	}
	isComputing_ = true;
}

void NeuralNetwork::finishComputation(bool sync) {
	require(isInitialized_);
	inputLayer_.finishComputation(sync);
	for (u32 l = 0; l < nLayer_; l++) {
		layer_.at(l)->finishComputation(sync);
	}
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->finishComputation(sync);
	}
	isComputing_ = false;
}
