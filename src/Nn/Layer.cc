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
 * Layer.cc
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#include "Layer.hh"
#include "ActivationLayer.hh"
#include "FeatureTransformationLayer.hh"
#include "MultiPortLayer.hh"
#include <sys/stat.h>

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
		width_(1),
		height_(1),
		nChannels_(0),
		nPorts_(Core::Configuration::config(paramNumberOfPorts_, prefix_)),
		dropoutProbability_(Core::Configuration::config(paramDropoutProbability_, prefix_)),
		useDropout_(dropoutProbability_ > 0),
		nTimeframes_(0),
		trainingMode_(false),
		isInitialized_(false),
		isComputing_(false)
{}

void BaseLayer::initialize(const std::string &basePath, const std::string& suffix, u32 maxMemory) {
	if (nUnits_ == 0) {
		std::cerr << "Error: 0 is an invalid value for the parameter " << prefix_ << ".number-of-units. Abort." << std::endl;
		exit(1);
	}
	activations_.resize(nPorts_);
	errorSignals_.resize(nPorts_);
	if (useDropout_)
		dropoutMasks_.resize(nPorts_);
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).setMaximalMemory(maxMemory);
		if (trainingMode_)
			errorSignals_.at(port).setMaximalMemory(maxMemory);
		if (useDropout_)
			dropoutMasks_.at(port).setMaximalMemory(maxMemory);
	}
	if (useDropout_)
		Core::Log::os("Use dropout probablility ") << dropoutProbability_ << " in layer " << name_;
	isInitialized_ = true;
}

void BaseLayer::setWidth(u32 port, u32 width) {
	require_gt(width, 0);
	require_lt(port, nInputPorts());
	width_ = width;
}

void BaseLayer::setHeight(u32 port, u32 height) {
	require_gt(height, 0);
	require_lt(port, nInputPorts());
	height_ =  height;
}

void BaseLayer::setNumberOfChannels(u32 port, u32 channels) {
	require_gt(channels, 0);
	require_lt(port, nInputPorts());
	nChannels_ = channels;
}

void BaseLayer::updateNumberOfUnits(u32 port) {
	nUnits_ = height_ * width_ * nChannels_;
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

void BaseLayer::addTimeframe(u32 minibatchSize) {
	nTimeframes_++;
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).addTimeframe(nInputUnits(port), minibatchSize);
		activations_.at(port).getLast().setToZero();
		if (trainingMode_) {
			errorSignals_.at(port).addTimeframe(nInputUnits(port), minibatchSize);
			errorSignals_.at(port).getLast().setToZero();
		}
		if (useDropout_) {
			dropoutMasks_.at(port).addTimeframe(nInputUnits(port), minibatchSize);
		}
	}
}

void BaseLayer::addEmptyTimeframe() {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).addTimeframe(0, 0);
		if (trainingMode_)
			errorSignals_.at(port).addTimeframe(0, 0);
		if (useDropout_)
			dropoutMasks_.at(port).addTimeframe(0, 0);
	}
	nTimeframes_++;
}

void BaseLayer::setMaximalMemory(u32 maxMemory) {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).setMaximalMemory(maxMemory);
		if (trainingMode_)
			errorSignals_.at(port).setMaximalMemory(maxMemory);
		if (useDropout_)
			dropoutMasks_.at(port).setMaximalMemory(maxMemory);
	}
}

void BaseLayer::resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns) {
	require_lt(timeframe, nTimeframes());
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).at(timeframe).resize(nRows, nColumns);
		if (trainingMode_)
			errorSignals_.at(port).at(timeframe).resize(nRows, nColumns);
		if (useDropout_)
			dropoutMasks_.at(port).at(timeframe).resize(nRows, nColumns);
	}
}

void BaseLayer::reset() {
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).reset();
		if (trainingMode_)
			errorSignals_.at(port).reset();
		if (useDropout_)
			dropoutMasks_.at(port).reset();
	}
	nTimeframes_ = 0;
}

void BaseLayer::setActivationVisibility(u32 timeframe, u32 nVisibleColumns) {
	require_lt(timeframe, nTimeframes());
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).at(timeframe).setVisibleColumns(nVisibleColumns);
		if (useDropout_)
			dropoutMasks_.at(port).at(timeframe).setVisibleColumns(nVisibleColumns);
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
			dropoutMasks_.at(port).at(t).fill(1.0);
			dropoutMasks_.at(port).at(t).dropout(dropoutProbability_);
			activationsOut(t, port).elementwiseMultiplication(dropoutMasks_.at(port).at(t));
			activationsOut(t, port).scale(1.0 / (1.0 - dropoutProbability_));
		}
	}
}

void BaseLayer::initComputation(bool sync) {
	if (isComputing_)
		return;
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).initComputation(sync);
		if (trainingMode_)
			errorSignals_.at(port).initComputation(sync);
		if (useDropout_)
			dropoutMasks_.at(port).initComputation(sync);
	}
	isComputing_ = true;
}

void BaseLayer::finishComputation(bool sync) {
	if (!isComputing_)
		return;
	for (u32 port = 0; port < nPorts_; port++) {
		activations_.at(port).finishComputation(sync);
		if (trainingMode_)
			errorSignals_.at(port).finishComputation(sync);
		if (useDropout_)
			dropoutMasks_.at(port).finishComputation(sync);
	}
	isComputing_ = false;
}

/*
 * Layer
 */
const Core::ParameterEnum Layer::paramLayerType_("type",
		"identity, sigmoid, tanh, softmax, max, exponential, logarithmic, rectified, triangle, clipped,"
		"l2-normalization, power-normalization, polynomial-preprocessing, fisher-encoding, sequence-length-normalization,"
		"feature-cloning, approximate-feature-map, modulated-sum, maxout, multiplication, temporal-averaging, concatenation, gated-recurrent-unit,"
		"attention, max-pooling, avg-pooling, batch-normalization, temporal-reversion, pre-processing",
		"identity", "neural-network.layer");

const Core::ParameterBool Layer::paramUseBias_("use-bias", true, "neural-network.layer");

const Core::ParameterBool Layer::paramIsBiasTrainable_("is-bias-trainable", true, "neural-network.layer");

const Core::ParameterEnum Layer::paramBiasInitialization_("bias-initialization", "random, zero", "random", "neural-network.layer");

const Core::ParameterFloat Layer::paramRandomBiasMin_("random-bias-min", -0.1, "neural-network.layer");

const Core::ParameterFloat Layer::paramRandomBiasMax_("random-bias-max", 0.1, "neural-network.layer");

const Core::ParameterBool Layer::paramUseCudnn_("use-cudnn", true, "neural-network.layer");

const Core::ParameterFloat Layer::paramLearningRateFactor_("learning-rate-factor", 1.0, "neural-network.layer");

Layer::Layer(const char* name) :
		Precursor(name),
		type_((LayerType) Core::Configuration::config(paramLayerType_, prefix_)),
		useBias_(Core::Configuration::config(paramUseBias_, prefix_)),
		isBiasTrainable_(Core::Configuration::config(paramIsBiasTrainable_, prefix_)),
		isRecurrent_(false),
		isOutputLayer_(false),
		useCudnn_(Core::Configuration::config(paramUseCudnn_, prefix_)),
		learningRateFactor_(Core::Configuration::config(paramLearningRateFactor_, prefix_))
{}

void Layer::initialize(const std::string &basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	if (useBias_) {
		initializeBias(basePath, suffix);
	}
	// make sure that every port has an incoming connection
	require_eq(nIncomingConnections_.size(), nInputPorts());
	for (u32 port = 0; port < nInputPorts(); port++) {
		if (nIncomingConnections_.at(port) < 1) {
			std::cerr << "Error: Port " << port << " of layer " << name_ << " has no incoming connection. Abort." << std::endl;
			exit(1);
		}
	}
	isInitialized_ = true;
}

void Layer::_initializeParam(Vector &param, ParamInitialization initMethod) {
	require(!isComputing_);

	Float min = Core::Configuration::config(paramRandomBiasMin_, prefix_);
	Float max = Core::Configuration::config(paramRandomBiasMax_, prefix_);
	require_lt(min, max);

	switch (initMethod) {
	case zero:
		param.setToZero();
		break;
	case random:
	default:
		for (u32 i = 0; i < param.nRows(); i++) {
			param.at(i) = Math::Random::random(min, max);
		}
		break;
	}
}

void Layer::_initializeBias(u32 port) {
	require(!isComputing_);
	require(useBias_);

	_initializeParam(bias_.at(port), (ParamInitialization)Core::Configuration::config(paramBiasInitialization_, prefix_));
}

std::string Layer::getParamFileName(const std::string& basePath, const std::string& paramName,
		const std::string& suffix, u32 port) {

	std::stringstream filename;
	filename << basePath << paramName << "-" << name_ << ".vector.gz";
	std::string fn = filename.str();

	std::stringstream sfx;
	sfx << suffix;
	if (nInputPorts() > 1)
		sfx << ".port-" << port;

	Core::Utils::appendSuffix(fn, sfx.str());

	return fn;
}

void Layer::initializeBias(const std::string& basePath, const std::string& suffix) {
	require(!isComputing_);
	require(useBias_);
	bias_.resize(nInputPorts());
	for (u32 port = 0; port < nInputPorts(); port++)
	{
		bias_.at(port).resize(nChannels(port));

		struct stat buffer;
		if (suffix.compare("") != 0 &&
				stat(getParamFileName(basePath, "bias", suffix, port).c_str(), &buffer) == 0) {
			std::string filename = getParamFileName(basePath, "bias", suffix, port);
			Core::Log::os("Layer ") << name_ << ":" << port << ": read bias from " << filename;
			bias_.at(port).read(filename);
		}
		else if (stat(getParamFileName(basePath, "bias", "", port).c_str(), &buffer) == 0) {
			std::string filename = getParamFileName(basePath, "bias", "", port);
			Core::Log::os("Layer ") << name_ << ":" << port << ": read bias from " << filename;
			bias_.at(port).read(filename);
		}
		else {
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
			_initializeBias(port);
		}
		require_eq(bias_.at(port).nRows(), nChannels(port));
	}
}
void Layer::saveParams(const std::string& basePath, const std::string& suffix) {
	if (useBias_) {
		saveBias(basePath, suffix);
	}
}
void Layer::saveBias(const std::string& basePath, const std::string& suffix) {
	require(useBias_);
	for (u32 port = 0; port < nInputPorts(); port++) {
		std::string fn = getParamFileName(basePath, "bias", suffix, port);

		Core::Log::os("Layer ") << name_ << ":port-" << port << ": write bias to " << fn;

		bool isBiasComputing = bias_.at(port).isComputing();
		bias_.at(port).finishComputation();
		bias_.at(port).write(fn);
		if (isBiasComputing)
			bias_.at(port).initComputation(false);
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
	if ( (!isRecurrent_) && (c->isRecurrent()) ) {
		isRecurrent_ = true;
	}
}

void Layer::addOutgoingConnection(Connection* c, u32 port) {
	require_lt(port, nOutputPorts());
	outgoingConnections_.resize(nOutputPorts());
	nOutgoingConnections_.resize(nOutputPorts(), 0);
	outgoingConnections_.at(port).push_back(c);
	nOutgoingConnections_.at(port)++;
	// check if the connection is recurrent
	if ( (!isRecurrent_) && (c->isRecurrent()) ) {
		isRecurrent_ = true;
	}
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
	//require(useBias_);
	require(isBiasTrainable());
	require(port < nInputPorts());
	return bias_.at(port);
}

void Layer::setAsOutputLayer() {
	for (u32 port = 0; port < nOutputPorts(); port++) {
		if (nOutgoingConnections(port) > 0) {
			for (u32 i = 0; i < nOutgoingConnections(port); i++) {
				if (!outgoingConnection(i, port).isRecurrent())
					Core::Error::msg("Layer ") << name()
					<< " can not be set as output layer and have non-recurrent outgoing connections at the same time." << Core::Error::abort;
			}
		}
	}
	isOutputLayer_ = true;
}

void Layer::forward(u32 port) {
	require(isComputing_);
	u32 t = nTimeframes() - 1;
	// input * weights for all incoming connections
	for (u32 i = 0; i < nIncomingConnections(port); i++) {
		incomingConnection(i, port).forwardWeightMultiplication();
	}
	if (useBias_) {
		// if no CNN: nChannels_ is the number of units
		activationsIn(t, port).addToAllChannels(bias_.at(port), nChannels(port));
	}
}

void Layer::forward() {
	for (u32 port = 0; port < nInputPorts(); port++)
		forward(port);
}

void Layer::backpropagate(u32 timeframe, u32 port) {
	require(isComputing_);
	require(trainingMode_);
	require_lt(timeframe, nTimeframes());
	for (u32 i = 0; i < nOutgoingConnections(port); i++) {
		outgoingConnection(i, port).backpropagateWeights(timeframe);
	}
	if (useDropout_) {
		errorSignalOut(timeframe, port).elementwiseMultiplication(dropoutMasks_.at(port).at(timeframe));
		errorSignalOut(timeframe, port).scale(1.0 / (1.0 - dropoutProbability_));
	}
}

void Layer::backpropagate(u32 timeframe) {
	for (u32 port = 0; port < nOutputPorts(); port++)
		backpropagate(timeframe, port);
}

void Layer::initComputation(bool sync) {
	if (isComputing_)
		return;
	Precursor::initComputation(sync);
	if (useBias_) {
		for (u32 port = 0; port < nInputPorts(); port++)
			bias_.at(port).initComputation(sync);
	}
	isComputing_ = true;
}

void Layer::finishComputation(bool sync) {
	if (!isComputing_)
		return;
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
	case max:
		Core::Log::os("Create max layer.");
		layer = new MaxLayer(name);
		break;
	case exponential:
		Core::Log::os("Create exponential layer.");
		layer = new ExponentialLayer(name);
		break;
	case logarithmic:
		Core::Log::os("Create logarithmic layer.");
		layer = new LogarithmicLayer(name);
		break;
	case rectified:
		Core::Log::os("Create rectified layer.");
		layer = new RectifiedLayer(name);
		break;
	case triangle:
		Core::Log::os("Create triangle layer.");
		layer = new TriangleActivationLayer(name);
		break;
	case clipped:
		Core::Log::os("Create clipped layer.");
		layer = new ClippedLayer(name);
		break;
	case l2normalization:
		Core::Log::os("Create l2-normalization layer.");
		layer = new L2NormalizationLayer(name);
		break;
	case powerNormalization:
		Core::Log::os("Create power-normalization layer.");
		layer = new PowerNormalizationLayer(name);
		break;
	case polynomialPreprocessing:
		Core::Log::os("Create polynomial-preprocessing layer.");
		layer = new PolynomialPreprocessingLayer(name);
		break;
	case fisherEncoding:
		Core::Log::os("Create fisher-encoding layer.");
		layer = new FisherLayer(name);
		break;
	case sequenceLengthNormalization:
		Core::Log::os("Create sequence-length-normalization layer.");
		layer = new SequenceLengthNormalizationLayer(name);
		break;
	case featureCloning:
		Core::Log::os("Create feature-cloning layer.");
		layer = new FeatureCloningLayer(name);
		break;
	case approximateFeatureMap:
		Core::Log::os("Create chi-square-feature-map layer.");
		layer = new ApproximateFeatureMapLayer(name);
		break;
	case modulatedSum:
		Core::Log::os("Create modulated-sum layer.");
		layer = new ModulatedSumLayer(name);
		break;
	case maxout:
		Core::Log::os("Create maxout layer.");
		layer = new MaxoutLayer(name);
		break;
	case multiplication:
		Core::Log::os("Create multiplication layer.");
		layer = new MultiplicationLayer(name);
		break;
	case temporalAveraging:
		Core::Log::os("Create temporal-averaging layer.");
		layer = new TemporalAveragingLayer(name);
		break;
	case concatenation:
		Core::Log::os("Create concatenation layer.");
		layer = new ConcatenationLayer(name);
		break;
	case gatedRecurrentUnit:
		Core::Log::os("Create gated-recurrent-unit layer");
		layer = new GatedRecurrentUnitLayer(name);
		break;
	case attention:
		Core::Log::os("Create attention layer");
		layer = new AttentionLayer(name);
		break;
	case maxPoolingLayer:
		Core::Log::os("Create max-pooling layer.");
		layer = new MaxPoolingLayer(name);
		break;
	case avgPoolingLayer:
		Core::Log::os("Create avg-pooling layer.");
		layer = new AvgPoolingLayer(name);
		break;
	case batchNormalizationLayer:
		Core::Log::os("Create batch-normalization layer.");
		layer = new BatchNormalizationLayer(name);
		break;
	case temporalReversion:
		Core::Log::os("Create temporal-reversion layer.");
		layer = new TemporalReversionLayer(name);
		break;
	case preProcessing:
		Core::Log::os("Create pre-processing layer.");
		layer = new PreProcessingLayer(name);
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

void InputLayer::initialize(u32 inputDimension, u32 width, u32 height, u32 nChannels) {
	nUnits_ = inputDimension;
	if ((height == 0) || (width == 0) || (nChannels == 0)) {
		nChannels_ = inputDimension;
		width_ = 1;
		height_ = 1;
	}
	else {
		nChannels_ = nChannels;
		height_ = height;
		width_ = width;
		require_eq(nUnits_, height_ * width_ * nChannels_);
	}

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
	// TODO: copy input?
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
