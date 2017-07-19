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
 * NeuralNetwork.cc
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#include "NeuralNetwork.hh"
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <map>

using namespace Nn;

const Core::ParameterStringList NeuralNetwork::paramConnections_("connections", "", "neural-network");

const Core::ParameterInt NeuralNetwork::paramInputDimension_("input-dimension", 0, "neural-network");

const Core::ParameterInt NeuralNetwork::paramSourceWidth_("source-width", 0, "neural-network");

const Core::ParameterInt NeuralNetwork::paramSourceHeight_("source-height", 0, "neural-network");

const Core::ParameterInt NeuralNetwork::paramSourceChannels_("source-channels", 0, "neural-network");

const Core::ParameterString NeuralNetwork::paramWriteParamsTo_("write-model-to", "", "neural-network");

const Core::ParameterString NeuralNetwork::paramLoadParamsFrom_("load-model-from", "", "neural-network");

const Core::ParameterInt NeuralNetwork::paramLoadParamsEpoch_("load-epoch", Types::max<s32>(), "neural-network");

NeuralNetwork::NeuralNetwork() :
		inputDimension_(Core::Configuration::config(paramInputDimension_)),
		sourceWidth_(Core::Configuration::config(paramSourceWidth_)),
		sourceHeight_(Core::Configuration::config(paramSourceHeight_)),
		sourceChannels_(Core::Configuration::config(paramSourceChannels_)),
		nLayer_(0),
		isRecurrent_(false),
		isInitialized_(false),
		isComputing_(false),
		writeParamsTo_(Core::Configuration::config(paramWriteParamsTo_)),
		loadParamsFrom_(Core::Configuration::config(paramLoadParamsFrom_)),
		loadParamsEpoch_(Core::Configuration::config(paramLoadParamsEpoch_))
{
	if (!writeParamsTo_.empty())
		writeParamsTo_.append("/");
	if (!loadParamsFrom_.empty())
		loadParamsFrom_.append("/");
}

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
	bool isRecurrent = true;
	if ((sourceLayer->name().compare("network-input") == 0) ||
			(layerNameToIndex_[sourceLayer->name()] < layerNameToIndex_[destLayer->name()])) {
		isRecurrent = false;
	}
	Connection* newConnection = Connection::createConnection(connectionName.c_str(), sourceLayer, destLayer, sourcePort, destPort, isRecurrent);
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

void NeuralNetwork::checkTopology() {
	/*
	 * topology is given by neural-network.connections
	 * check that each layer has a topologically preceding layer
	 */
	std::vector<BaseLayer*> checkedLayers;
	checkedLayers.push_back(&inputLayer_);
	for (u32 l = 0; l < layer_.size(); l++) {
		bool hasPredecessor = false;
		for (u32 port = 0; port < layer_.at(l)->nInputPorts(); port++) {
			for (u32 c = 0; c < layer_.at(l)->nIncomingConnections(port); c++) {
				if (containsLayer(&(layer_.at(l)->incomingConnection(c, port).from()), checkedLayers)) {
					hasPredecessor = true;
				}
			}
		}
		if (!hasPredecessor) {
			Core::Error::msg("Layer ") << layer_.at(l)->name()
					<< " has no predecessor is the topological order specified by neural-network.connections." << Core::Error::abort;
		}
		checkedLayers.push_back((BaseLayer*)layer_.at(l));
	}
	Core::Log::openTag("topological-order");
	for (u32 l = 0; l < layer_.size(); l++) {
		Core::Log::os() << layer_.at(l)->name();
	}
	Core::Log::closeTag();
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
	if (inputDimension_ == 0)
		Core::Error::msg("Error: 0 is an invalid value for the parameter neural-network.input-dimension.") << Core::Error::abort;
	buildNetwork();
	checkTopology();
	// output layer has exactly one output port
	if (layer(nLayer_ - 1).nOutputPorts() != 1)
		Core::Error::msg("Error: Output layer ") << layer(nLayer_ - 1).name() << " must not have more than one output port." << Core::Error::abort;
	logTopology();
	//initialize input layer
	inputLayer_.initialize(inputDimension_, sourceWidth_, sourceHeight_, sourceChannels_);
	// first initialize all connections...
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->initialize();
	}
	// ... then initialize all layers...
	std::stringstream s;
	s.clear();
	if ((s32)loadParamsEpoch_ < Types::max<s32>()) {
		s << ".epoch-" << loadParamsEpoch_;
	}
	for (u32 l = 0; l < nLayer_; l++) {
		layer_.at(l)->initialize(loadParamsFrom_, s.str(), maxMemory);
	}
	// ... and initialize the weights of the connections
	for (u32 c = 0; c < connections_.size(); c++) {
		connections_.at(c)->initializeWeights(loadParamsFrom_, s.str());
	}
	Core::Log::closeTag();
	// by default, set the last layer as output layer
	layer_.back()->setAsOutputLayer();
	isInitialized_ = true;
}

bool NeuralNetwork::requiresFullMemorization() const {
	for (u32 l = 0; l < nLayer(); l++) {
		if (layer(l).requiresFullPass())
			return true;
	}
	return false;
}

u32 NeuralNetwork::lastRecurrentLayerIndex() const {
	u32 index = 0;
	for (u32 l = 0; l < nLayer_; l++) {
		// layers requiring a full pass are also relevant here
		index = ( (layer(l).isRecurrent() || layer(l).requiresFullPass()) ? l : index);
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

Layer& NeuralNetwork::layer(std::string& name) const {
	if (layerNameToIndex_.find(name) == layerNameToIndex_.end()) {
		Core::Error::msg("NeuralNetwork::layer: can not find layer with name ") << name << Core::Error::abort;
	}
	return *(layer_.at(layerNameToIndex_.at(name)));
}

Connection& NeuralNetwork::connection(u32 connectionIndex) const {
	require(isInitialized_);
	require(connectionIndex < connections_.size());
	return *(connections_.at(connectionIndex));
}

void NeuralNetwork::forward(Matrix& batch, u32 layerIndexFrom, u32 layerIndexTo) {
	require(isInitialized_);
	layerIndexTo = std::min(nLayer_ - 1, layerIndexTo);
	if (layerIndexFrom == 0) {
		// reset all activations
		reset();
		inputLayer_.addInput(batch);
		initComputation();
	}

	// forward all layers
	for (u32 l = layerIndexFrom; l <= layerIndexTo; l++) {
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

void NeuralNetwork::forwardTimeframe(MatrixContainer& batchedSequence, u32 t, u32 layerIndexFrom, u32 layerIndexTo, bool greedyForwarding) {
	layerIndexTo = std::min(nLayer_ - 1, layerIndexTo);
	// number of active sequences at timeframe t+1
	u32 size = (t == batchedSequence.nTimeframes() - 1 ?
			batchedSequence.getLast().nColumns() : batchedSequence.at(t+1).nColumns());
	if (layerIndexFrom == 0) {
		// add the input
		inputLayer_.addInput(batchedSequence.at(t));
		initComputation();
	}

	u32 forwardTo = layerIndexTo;
	// greedy forwarding: forward layers up to last recurrent layer (or the last layer that requires a full pass)
	//                    if final timeframe not yet reached, else forward all layers
	if (greedyForwarding && (t < batchedSequence.nTimeframes() - 1)) {
		// if layerIndexTo does not require a full pass (i.e. is last layer), maybe greedy forwarding can be is desired here
		forwardTo = std::min(layerIndexTo, lastRecurrentLayerIndex());
	}

	/* prepare sequence forwarding */
	for (u32 l = layerIndexFrom; l <= forwardTo; l++) {
		// allocate storage for one time frame in the activations container...
		layer(l).addTimeframe(size);
		// ... hide the elements that are currently not needed ...
		layer(l).setActivationVisibility(t, batchedSequence.at(t).nColumns());
		if (layer(l).isInTrainingMode())
			layer(l).setErrorSignalVisibility(t, batchedSequence.at(t).nColumns());
		// ... unhide the elements from the previous time frame that may be needed due to recurrency...
		if ((t > 0) && layer(l).isRecurrent())
			layer(l).setActivationVisibility(t-1, batchedSequence.at(t).nColumns());
	}
	for (u32 l = layerIndexFrom; l <= forwardTo; l++) {
		/* forward the input */
		layer(l).forward();
		// apply dropout if desired
		if (layer(l).useDropout())
			layer(l).dropout();
	}
	for (u32 l = layerIndexFrom; l <= forwardTo; l++) {
		/* finalize forwarding */
		// hide the elements that have been unhidden
		if ((t > 0) && layer(l).isRecurrent())
			layer(l).setActivationVisibility(t-1, batchedSequence.at(t-1).nColumns());
		// finalize the forwarding if the last time frame is reached
		if (t == batchedSequence.nTimeframes() - 1)
			layer(l).finalizeForwarding();
	}
	// add empty timeframes to every layer that has not been forwarded due to greedy forwarding
	// empty timeframes do not allocate memory but just keep the timeframe numbering consistent
	for (u32 l = forwardTo + 1; l <= layerIndexTo; l++) {
		layer(l).addEmptyTimeframe();
	}
}

void NeuralNetwork::forwardSequence(MatrixContainer& batchedSequence, bool greedyForwarding) {
	require(isInitialized_);
	reset();
	// forward up to the next layer that requires a full forward pass over all timeframes until the last layer is reached
	u32 layerIndexFrom = 0;
	while (layerIndexFrom < nLayer()) {
		u32 layerIndexTo = layerIndexFrom;
		// get layer up to which the forwarding can be done
		while ((layerIndexTo < nLayer()-1) && (!layer_.at(layerIndexTo)->requiresFullPass()))
			layerIndexTo++;
		// forward all timeframes for the block [layerIndexFrom, layerIndexTo]
		for (u32 t = 0; t < batchedSequence.nTimeframes(); t++)
			forwardTimeframe(batchedSequence, t, layerIndexFrom, layerIndexTo, greedyForwarding);
		// set the values for the next block
		layerIndexFrom = layerIndexTo + 1;
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
	if (!writeParamsTo_.empty()) {
		for (u32 l = 0; l < layer_.size(); l++) {
			layer_.at(l)->saveParams(writeParamsTo_, suffix);
		}
		for (u32 c = 0; c < connections_.size(); c++) {
			connections_.at(c)->saveWeights(writeParamsTo_, suffix);
		}
	}
}

void NeuralNetwork::initComputation(bool sync) {
	require(isInitialized_);
	inputLayer_.initComputation(sync);
	if (!isComputing_) {
		for (u32 l = 0; l < nLayer_; l++) {
			layer_.at(l)->initComputation(sync);
		}
		for (u32 c = 0; c < connections_.size(); c++) {
			connections_.at(c)->initComputation(sync);
		}
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
