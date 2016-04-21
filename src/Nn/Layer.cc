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

#include "Layer.hh"
#include "ActivationLayer.hh"
#include "PortFusionLayer.hh"

using namespace Nn;

/*
 * BaseLayer
 */
const Core::ParameterInt BaseLayer::paramNumberOfUnits_("number-of-units", 0, "neural-network.layer");

const Core::ParameterInt BaseLayer::paramNumberOfPorts_("number-of-ports", 1, "neural-network.layer");

const Core::ParameterFloat BaseLayer::paramDropoutProbability_("dropout-probability", 0.0, "neural-network.layer");

BaseLayer::BaseLayer(const char* name) :
		name_(name),
		prefix_(std::string("neural-network.").append(name_)),
		nUnits_(Core::Configuration::config(paramNumberOfUnits_, prefix_)),
		nPorts_(Core::Configuration::config(paramNumberOfPorts_, prefix_)),
		dropoutProbability_(Core::Configuration::config(paramDropoutProbability_, prefix_)),
		useDropout_(dropoutProbability_ > 0),
		nTimeframes_(0),
		trainingMode_(false),
		isInitialized_(false),
		isComputing_(false)
{}

void BaseLayer::initialize(u32 maxMemory) {
	if (nUnits_ == 0) {
		std::cerr << "Error: 0 is an invalid value for the parameter " << prefix_ << ".number-of-units. Abort." << std::endl;
		exit(1);
	}
	activations_.resize(nPorts_);
	errorSignals_.resize(nPorts_);
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).setMaximalMemory(maxMemory);
		if (trainingMode_)
			errorSignals_.at(port).setMaximalMemory(maxMemory);
	}
	if (useDropout_)
		Core::Log::os("Use dropout probablility ") << dropoutProbability_ << " in layer " << name_;
	isInitialized_ = true;
}

void BaseLayer::setTrainingMode(bool trainingMode) {
	trainingMode_ = trainingMode;
}

Matrix& BaseLayer::activationsIn(u32 timeframe, u32 port) {
	require_lt(port, nInputPorts());
	require_lt(timeframe, activations_.at(port).nTimeframes());
	return activations_.at(port).at(timeframe);
}

Matrix& BaseLayer::latestActivations(u32 port) {
	require_gt(nTimeframes(), 0);
	return activations(nTimeframes() - 1, port);
}

Matrix& BaseLayer::errorSignalIn(u32 timeframe, u32 port) {
	require(trainingMode_);
	require_lt(port, nInputPorts());
	require_lt(timeframe, errorSignals_.at(port).nTimeframes());
	return errorSignals_.at(port).at(timeframe);
}

Matrix& BaseLayer::latestErrorSignal(u32 port) {
	require(trainingMode_);
	require_gt(nTimeframes(), 0);
	return errorSignal(nTimeframes() - 1, port);
}

void BaseLayer::addTimeframe(u32 minibatchSize, bool initWithZero) {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).addTimeframe(nUnits_, minibatchSize);
		if (initWithZero)
			activations_.at(port).getLast().setToZero();
		if (trainingMode_) {
			errorSignals_.at(port).addTimeframe(nUnits_, minibatchSize);
			if (initWithZero)
				errorSignals_.at(port).getLast().setToZero();
		}
	}
	nTimeframes_++;
}

void BaseLayer::addEmptyTimeframe() {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).addTimeframe(0, 0);
		if (trainingMode_)
			errorSignals_.at(port).addTimeframe(0, 0);
	}
	nTimeframes_++;
}

void BaseLayer::setMaximalMemory(u32 maxMemory) {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).setMaximalMemory(maxMemory);
		if (trainingMode_)
			errorSignals_.at(port).setMaximalMemory(maxMemory);
	}
}

void BaseLayer::resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns) {
	require_lt(timeframe, nTimeframes());
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).at(timeframe).resize(nRows, nColumns);
		if (trainingMode_)
			errorSignals_.at(port).at(timeframe).resize(nRows, nColumns);
	}
}

void BaseLayer::reset() {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).reset();
		if (trainingMode_)
			errorSignals_.at(port).reset();
	}
	nTimeframes_ = 0;
}

void BaseLayer::setActivationVisibility(u32 timeframe, u32 nVisibleColumns) {
	require_lt(timeframe, nTimeframes());
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).at(timeframe).setVisibleColumns(nVisibleColumns);
	}
}

void BaseLayer::setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns) {
	require(trainingMode_);
	require_lt(timeframe, nTimeframes());
	for (u32 port = 0; port < nPorts_; port++) {
		errorSignals_.at(port).at(timeframe).setVisibleColumns(nVisibleColumns);
	}
}

void BaseLayer::dropout() {
	u32 t = nTimeframes() - 1;
	if (useDropout_) {
		for (u32 port = 0; port < nOutputPorts(); port++) {
			activationsOut(t, port).dropout(dropoutProbability_);
		}
	}
}

void BaseLayer::initComputation(bool sync) {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).initComputation(sync);
		if (trainingMode_)
			errorSignals_.at(port).initComputation(sync);
	}
	isComputing_ = true;
}

void BaseLayer::finishComputation(bool sync) {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).finishComputation(sync);
		if (trainingMode_)
			errorSignals_.at(port).finishComputation(sync);
	}
	isComputing_ = false;
}

/*
 * Layer
 */
const Core::ParameterEnum Layer::paramLayerType_("type",
		"identity, sigmoid, tanh, softmax, rectified, l2-normalization, power-normalization, "
		"sequence-length-normalization, maxout, gated-recurrent-unit",
		"identity", "neural-network.layer");

const Core::ParameterBool Layer::paramUseBias_("use-bias", true, "neural-network.layer");

const Core::ParameterBool Layer::paramIsBiasTrainable_("is-bias-trainable", true, "neural-network.layer");

const Core::ParameterEnum Layer::paramBiasInitialization_("bias-initialization", "random, zero", "random", "neural-network.layer");

const Core::ParameterFloat Layer::paramRandomBiasMin_("random-bias-min", -0.1, "neural-network.layer");

const Core::ParameterFloat Layer::paramRandomBiasMax_("random-bias-max", 0.1, "neural-network.layer");

const Core::ParameterString Layer::paramOldBiasFilename_("load-bias-from", "", "neural-network.layer");

const Core::ParameterString Layer::paramNewBiasFilename_("write-bias-to", "", "neural-network.layer");

Layer::Layer(const char* name) :
		Precursor(name),
		type_((LayerType) Core::Configuration::config(paramLayerType_, prefix_)),
		useBias_(Core::Configuration::config(paramUseBias_, prefix_)),
		isBiasTrainable_(Core::Configuration::config(paramIsBiasTrainable_, prefix_)),
		isRecurrent_(false)
{}

void Layer::initialize(u32 maxMemory) {
	Precursor::initialize(maxMemory);
	if (useBias_) {
		initializeBias();
	}
	// make sure that every port has an incoming connection
	require_eq(nIncomingConnections_.size(), nInputPorts());
	for (u32 port = 0; port < nInputPorts(); port++) {
		if (nIncomingConnections_.at(port) < 1) {
			std::cerr << "Error: Port " << port << " of layer " << name_ << " has no incoming connection. Abort." << std::endl;
			exit(1);
		}
	}
	// make sure that every port has an outgoing connection if there is more than one port
	if (nOutputPorts() > 1) {
		require_eq(nOutgoingConnections_.size(), nOutputPorts());
		for (u32 port = 0; port < nOutputPorts(); port++) {
			if (nOutgoingConnections_.at(port) < 1) {
				std::cerr << "Error: Port " << port << " of layer " << name_ << " has no outgoing connection. Abort." << std::endl;
				exit(1);
			}
		}
	}
	isInitialized_ = true;
}

void Layer::_initializeBias() {
	require(!isComputing_);
	require(useBias_);
	Float min = Core::Configuration::config(paramRandomBiasMin_, prefix_);
	Float max = Core::Configuration::config(paramRandomBiasMax_, prefix_);
	require_lt(min, max);
	switch (Core::Configuration::config(paramBiasInitialization_, prefix_)) {
	case zero:
		for (u32 port = 0; port < nInputPorts(); port++) {
			bias_.at(port).setToZero();
		}
		break;
	case random:
		Math::RandomNumberGenerator rand;
		for (u32 port = 0; port < nInputPorts(); port++) {
			for (u32 i = 0; i < bias_.at(port).nRows(); i++) {
				bias_.at(port).at(i) = rand.random(min, max);
			}
		}
		break;
	default: // cannot happen
		break;
	}
}

void Layer::initializeBias() {
	require(!isComputing_);
	require(useBias_);
	bias_.resize(nInputPorts());
	for (u32 port = 0; port < nInputPorts(); port++)
		bias_.at(port).resize(nUnits_);
	std::string filename = Core::Configuration::config(paramOldBiasFilename_, prefix_);
	if (filename.empty()) {
		std::string initializationMethod;
		switch (Core::Configuration::config(paramBiasInitialization_, prefix_)) {
		case zero:
			initializationMethod = "zero";
			break;
		case random:
		default:
			initializationMethod = "random";
			break;
		}
		Core::Log::os("Layer ") << name_
				<< ": no file to load bias from. Use initialization: "
				<< initializationMethod;
		_initializeBias();
	}
	else {
		for (u32 port = 0; port < nInputPorts(); port++) {
			std::string fn = filename;
			std::stringstream suffix;
			if (nInputPorts() > 1)
				suffix << ".port-" << port;
			Core::Utils::appendSuffix(fn, suffix.str());
			Core::Log::os("Layer ") << name_ << ":" << port << ": read bias from " << fn;
			bias_.at(port).read(fn);
			require_eq(bias_.at(port).nRows(), nUnits_);
		}
	}
}

void Layer::saveBias(const std::string& suffix) {
	require(useBias_);
	std::string filename = Core::Configuration::config(paramNewBiasFilename_, prefix_);
	if (filename.empty()) {
		Core::Log::os("Layer ") << name_ << ": no file to write bias to specified. Do not save bias.";
	}
	else {
		for (u32 port = 0; port < nInputPorts(); port++) {
			std::string fn = filename;
			std::stringstream sfx;
			sfx << suffix;
			if (nInputPorts() > 1)
				sfx << ".port-" << port;
			Core::Utils::appendSuffix(fn, sfx.str());
			Core::Log::os("Layer ") << name_ << ":port-" << port << ": write bias to " << fn;
			bool isBiasComputing = bias_.at(port).isComputing();
			bias_.at(port).finishComputation();
			bias_.at(port).write(fn);
			if (isBiasComputing)
				bias_.at(port).initComputation(false);
		}
	}
}

u32 Layer::nIncomingConnections(u32 port) {
	require_lt(port, nInputPorts());
	nIncomingConnections_.resize(nInputPorts(), 0);
	return nIncomingConnections_.at(port);
}

u32 Layer::nOutgoingConnections(u32 port) {
	require_lt(port, nOutputPorts());
	nOutgoingConnections_.resize(nOutputPorts(), 0);
	return nOutgoingConnections_.at(port);
}

void Layer::addIncomingConnection(Connection* c, u32 port) {
	require_lt(port, nInputPorts());
	incomingConnections_.resize(nInputPorts());
	nIncomingConnections_.resize(nInputPorts(), 0);
	incomingConnections_.at(port).push_back(c);
	nIncomingConnections_.at(port)++;
	// check if the connection is recurrent
	if ( (!isRecurrent_) && (&(c->from()) == &(c->to())) ) {
		isRecurrent_ = true;
	}
}

void Layer::addOutgoingConnection(Connection* c, u32 port) {
	require_lt(port, nOutputPorts());
	outgoingConnections_.resize(nOutputPorts());
	nOutgoingConnections_.resize(nOutputPorts(), 0);
	outgoingConnections_.at(port).push_back(c);
	nOutgoingConnections_.at(port)++;
}

Connection& Layer::incomingConnection(u32 incomingConnectionIndex, u32 port) {
	require_lt(port, nIncomingConnections_.size());
	require_lt(incomingConnectionIndex, nIncomingConnections_.at(port));
	return *(incomingConnections_.at(port).at(incomingConnectionIndex));
}

Connection& Layer::outgoingConnection(u32 outgoingConnectionIndex, u32 port) {
	require_lt(port, nOutgoingConnections_.size());
	require_lt(outgoingConnectionIndex, nOutgoingConnections_.at(port));
	return *(outgoingConnections_.at(port).at(outgoingConnectionIndex));
}

bool Layer::isTrainable(u32 incomingConnectionIndex, u32 port) const {
	require_lt(port, nIncomingConnections_.size());
	require_lt(incomingConnectionIndex, nIncomingConnections_.at(port));
	return incomingConnections_.at(port).at(incomingConnectionIndex)->isTrainable();
}

Matrix& Layer::weights(u32 incomingConnectionIndex, u32 port) {
	require_lt(port, nIncomingConnections_.size());
	require_lt(incomingConnectionIndex, nIncomingConnections_.at(port));
	require(isTrainable(incomingConnectionIndex, port));
	WeightConnection* c = dynamic_cast<WeightConnection*>(incomingConnections_.at(port).at(incomingConnectionIndex));
	return c->weights();
}

Vector& Layer::bias(u32 port) {
	require(useBias_);
	require(port < nInputPorts());
	return bias_.at(port);
}

void Layer::forward(u32 port) {
	require(isComputing_);
	u32 t = nTimeframes() - 1;
	// set the activation to zero
	activationsIn(t, port).setToZero();
	// input * weights for all incoming connections
	for (u32 i = 0; i < nIncomingConnections(port); i++) {
		incomingConnection(i, port).forwardWeightMultiplication();
	}
	if (useBias_)
		activationsIn(t, port).addToAllColumns(bias_.at(port));
}

void Layer::forward() {
	for (u32 port = 0; port < nInputPorts(); port++)
		forward(port);
}

void Layer::backpropagate(u32 timeframe, u32 port) {
	require(isComputing_);
	require(trainingMode_);
	require_lt(timeframe, nTimeframes());
	// set the error signal to zero
	errorSignalOut(timeframe, port).setToZero();
	for (u32 i = 0; i < nOutgoingConnections(port); i++) {
		outgoingConnection(i, port).backpropagateWeights(timeframe);
	}
}

void Layer::backpropagate(u32 timeframe) {
	for (u32 port = 0; port < nOutputPorts(); port++)
		backpropagate(timeframe, port);
}

void Layer::initComputation(bool sync) {
	Precursor::initComputation(sync);
	if (useBias_) {
		for (u32 port = 0; port < nInputPorts(); port++)
			bias_.at(port).initComputation(sync);
	}
	isComputing_ = true;
}

void Layer::finishComputation(bool sync) {
	Precursor::finishComputation(sync);
	if (useBias_) {
		for (u32 port = 0; port < nInputPorts(); port++)
			bias_.at(port).finishComputation(sync);
	}
	isComputing_ = false;
}

/* Layer factory */
Layer* Layer::createLayer(const char* name) {
	Layer* layer = 0;
	std::string prefix("neural-network.");
	prefix.append(name);
	switch (Core::Configuration::config(paramLayerType_, prefix)) {
	case sigmoid:
		Core::Log::os("Create sigmoid layer.");
		layer = new SigmoidLayer(name);
		break;
	case tanh:
		Core::Log::os("Create tanh layer.");
		layer = new TanhLayer(name);
		break;
	case softmax:
		Core::Log::os("Create softmax layer.");
		layer = new SoftmaxLayer(name);
		break;
	case rectified:
		Core::Log::os("Create rectified layer.");
		layer = new RectifiedLayer(name);
		break;
	case l2normalization:
		Core::Log::os("Create l2-normalization layer.");
		layer = new L2NormalizationLayer(name);
		break;
	case powerNormalization:
		Core::Log::os("Create power-normalization layer.");
		layer = new PowerNormalizationLayer(name);
		break;
	case sequenceLengthNormalization:
		Core::Log::os("Create sequence-length-normalization layer.");
		layer = new SequenceLengthNormalizationLayer(name);
		break;
	case maxout:
		Core::Log::os("Create maxout layer.");
		layer = new MaxoutLayer(name);
		break;
	case gatedRecurrentUnit:
		Core::Log::os("Create gated-recurrent-unit layer");
		layer = new GatedRecurrentUnitLayer(name);
		break;
	case identity:
	default:
		Core::Log::os("Create identity layer.");
		layer = new Layer(name);
		break;
	}
	return layer;
}

/*
 * InputLayer
 */

InputLayer::InputLayer() :
		Precursor("network-input")
{}

void InputLayer::initialize(u32 inputDimension) {
	nUnits_ = inputDimension;
	if (useDropout_)
		Core::Log::os("Use dropout probablility ") << dropoutProbability_ << " for network input.";
	isInitialized_ = true;
}

void InputLayer::addInput(Matrix& input) {
	require_eq(input.nRows(), nUnits_);
	if (useDropout_) {
		bool computing = input.isComputing();
		if (!computing)
			input.initComputation();
		input.dropout(dropoutProbability_);
		if (!computing)
			input.finishComputation();
	}
	input_.push_back(&input);
	isComputing_ = isComputing_ && input.isComputing();
}

void InputLayer::reset() {
	input_.clear();
}

Matrix& InputLayer::activationsOut(u32 timeframe, u32 port) {
	require_lt(timeframe, input_.size());
	return *(input_.at(timeframe));
}

void InputLayer::initComputation(bool sync) {
	for (u32 i = 0; i < input_.size(); i++) {
		if (input_.at(i)) {
			input_.at(i)->initComputation(sync);
		}
	}
	isComputing_ = true;
}

void InputLayer::finishComputation(bool sync) {
	for (u32 i = 0; i < input_.size(); i++) {
		if (input_.at(i)) {
			input_.at(i)->finishComputation(sync);
		}
	}
	isComputing_ = false;
}
