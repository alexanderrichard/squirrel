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
 * Connection.cc
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#include "Connection.hh"
#include "Layer.hh"
#include "MatrixContainer.hh"
#include <Math/Random.hh>
#include <sstream>
#include <sys/stat.h>
using namespace Nn;

/*
 * Connection
 */

const Core::ParameterEnum Connection::paramConnectionType_("type",
		"plain-connection, weight-connection, convolutional-connection, valid-convolutional-connection",
		"weight-connection", "neural-network.connection");

const Core::ParameterFloat Connection::paramLearningRateFactor_("learning-rate-factor", 1.0, "neural-network.connection");

const Core::ParameterBool Connection::paramIsRecurrent_("is-recurrent", false, "neural-network.connection");

Connection::Connection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type) :
		name_(name),
		prefix_(std::string("neural-network.").append(name)),
		source_(source),
		dest_(dest),
		sourcePort_(sourcePort),
		destPort_(destPort),
		isRecurrent_(isRecurrent || Core::Configuration::config(paramIsRecurrent_, prefix_)),
		isComputing_(false),
		connectionType_(type),
		learningRateFactor_(Core::Configuration::config(paramLearningRateFactor_, prefix_))
{}

std::string Connection::getParamFileName(const std::string& basePath, const std::string& suffix) {
	std::stringstream filename;
	filename<<basePath<<"weights-"<<name_<<".matrix.gz";

	std::string fn = filename.str();
	Core::Utils::appendSuffix(fn, suffix);

	return fn;
}

bool Connection::isRecurrent() const {
	return isRecurrent_;
}

BaseLayer& Connection::from() {
	require(source_);
	return *(source_);
}

BaseLayer& Connection::to() {
	require(dest_);
	return *(dest_);
}

void Connection::_forwardWeightMultiplication(const Matrix& source, Matrix& dest) {
	require_eq(source.nRows(), dest.nRows());
	require_eq(source.nColumns(), dest.nColumns());
	dest.add(source);
}

void Connection::forwardWeightMultiplication() {
	require(isComputing_);
	require(source_);
	require(dest_);
	u32 t = dest_->nTimeframes() - 1; // the latest time frame index

	// note: dest_->activations(t) is not implicitly reset
	if (isRecurrent() && (dest_->nTimeframes() > 1)) {
		require_gt(source_->nTimeframes(), t-1);
		_forwardWeightMultiplication(source_->activationsOut(t-1, sourcePort_), dest_->activationsIn(t, destPort_));
	}
	else {
		require_gt(source_->nTimeframes(), t);
		_forwardWeightMultiplication(source_->activationsOut(t, sourcePort_), dest_->activationsIn(t, destPort_));
	}
}

void Connection::_backpropagateWeights(const Matrix& source, Matrix& dest) {
	_forwardWeightMultiplication(source, dest);
}

void Connection::backpropagateWeights(u32 timeframe) {
	require(isComputing_);
	require(source_);
	require(dest_);
	require_eq(source_->nTimeframes(), dest_->nTimeframes());
	// default behavior: just backpropagate the old error signal (equivalent to weight-matrix = identity)
	if (isRecurrent() && (timeframe + 1 < dest_->nTimeframes())) {
		_backpropagateWeights(dest_->errorSignalIn(timeframe + 1, destPort_), source_->errorSignalOut(timeframe, sourcePort_));
	}
	// if not recurrent and error signal of dest_ at (timeframe, destPort) exists, backpropagate
	// error signal might not exists if dest_ is beyond the last recurrent layer
	else if ((!isRecurrent()) && (dest_->errorSignalIn(timeframe, destPort_).nRows() > 0)) {
		_backpropagateWeights(dest_->errorSignalIn(timeframe, destPort_), source_->errorSignalOut(timeframe, sourcePort_));
	}
}

void Connection::setWeightsFileSuffix() {
	std::stringstream s;
	s << "." << sourcePort_ << "-" << destPort_;
	weightsFileSuffix_ = s.str();
	name_.append(s.str());
}

/*
 * Connection factory
 */
Connection* Connection::createConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent) {
	Connection* connection = 0;
	std::string prefix("neural-network.");
	prefix.append(name);
	ConnectionType type = (ConnectionType) Core::Configuration::config(paramConnectionType_, prefix);
	switch (type) {
	case weightConnection:
		connection = new WeightConnection(name, source, dest, sourcePort, destPort, isRecurrent, type);
		break;
	case convolutionalConnection:
		Core::Log::os("Create Convolutional Connection.");
		connection = new ConvolutionalConnection(name, source, dest, sourcePort, destPort, isRecurrent, type);
		break;
	case plainConnection:
		Core::Log::os("Create Plain Connection.");
		connection = new PlainConnection(name, source, dest, sourcePort, destPort, isRecurrent,type);
		break;
	case validConvolutionalConnection:
		Core::Log::os("Create Valid Convolutional Connection.");
		connection =  new ValidConvolutionalConnection(name, source, dest, sourcePort, destPort, isRecurrent, type);
		break;
	default: // cannot happen
		break;
	}
	return connection;
}

/*
 * PlainConnection
 */

PlainConnection::PlainConnection(const char* name, BaseLayer *source, BaseLayer *dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type):
		Precursor(name, source, dest, sourcePort, destPort, isRecurrent, type)
{}

void PlainConnection::initialize() {
	Precursor::initialize();
	to().setHeight(destPort_, from().height(sourcePort_));
	to().setWidth(destPort_, from().width(sourcePort_));
	to().setNumberOfChannels(destPort_, from().nChannels(sourcePort_));
	to().updateNumberOfUnits(destPort_);
}

/*
 * WeightConnection
 */
const Core::ParameterBool WeightConnection::paramIsTrainable_("is-trainable", true, "neural-network.connection");

const Core::ParameterEnum WeightConnection::paramWeightInitialization_("weight-initialization", "random, zero, identity, glorot",
		"random", "neural-network.connection");

const Core::ParameterFloat WeightConnection::paramRandomWeightMin_("random-weight-min", -0.1, "neural-network.connection");

const Core::ParameterFloat WeightConnection::paramRandomWeightMax_("random-weight-max", 0.1, "neural-network.connection");

WeightConnection::WeightConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type) :
		Precursor(name, source, dest, sourcePort, destPort, isRecurrent, type),
		isTrainable_(Core::Configuration::config(paramIsTrainable_, prefix_))
{}

void WeightConnection::initialize() {
	Precursor::initialize();
	to().setNumberOfChannels(destPort_, to().nInputUnits(destPort_) > 0 ? to().nInputUnits(destPort_) : 1);
}

void WeightConnection::_initializeWeights(u32 nRows, u32 nColumns) {
	require(!isComputing_);
	weights_.resize(nRows, nColumns);
	switch (Core::Configuration::config(paramWeightInitialization_, prefix_)) {
	case zero:
		Core::Log::os("Connection ") << name_ << ": no file to load weights from. Use zero initialization.";
		weights_.setToZero();
		break;
	case identity:
		Core::Log::os("Connection ") << name_ << ": no file to load weights from. Use identity initialization.";
		weights_.setToZero();
		for (u32 i = 0; i < std::min(weights_.nRows(), weights_.nColumns()); i++)
			weights_.at(i,i) = 1;
		break;
	case random:
	{
		Core::Log::os("Connection ") << name_ << ": no file to load weights from. Use random initialization.";
		Float min = Core::Configuration::config(paramRandomWeightMin_, prefix_);
		Float max = Core::Configuration::config(paramRandomWeightMax_, prefix_);
		require_lt(min, max);
		for (u32 i = 0; i < weights_.nRows(); i++) {
			for (u32 j = 0; j < weights_.nColumns(); j++) {
				weights_.at(i,j) = Math::Random::random(min, max);
			}
		}
	}
		break;
	case glorot:
	{
		Core::Log::os("Connection ") << name_ << ": no file to load weights from. Use glorot initialization.";
		Float min = -1.0f/sqrt(from().nOutputUnits(sourcePort_));
		Float max = 1.0f/sqrt(from().nOutputUnits(sourcePort_));
		for (u32 i = 0; i < weights_.nRows(); i++) {
			for (u32 j = 0; j < weights_.nColumns(); j++) {
				weights_.at(i,j) = Math::Random::random(min, max);
			}
		}
	}
		break;
	default: // cannot happen
		break;
	}
}

void WeightConnection::_initializeWeights(const std::string& basePath, const std::string& suffix, u32 nRows, u32 nColumns) {
	struct stat buffer;
	if (suffix.compare("") != 0 && stat(getParamFileName(basePath, suffix).c_str(), &buffer) == 0) {
		std::string filename = getParamFileName(basePath, suffix);
		Core::Log::os("Connection ") << name_ << ": read weight matrix from " << filename;
		weights_.read(filename);
	}
	else if (stat(getParamFileName(basePath, "").c_str(), &buffer) == 0) {
		std::string filename = getParamFileName(basePath, "");
		Core::Log::os("Connection ") << name_ << ": read weight matrix from " << filename;
		weights_.read(filename);
	}
	else {
		std::string filename = getParamFileName(basePath, "");
		Core::Log::os("Connection ") << name_ << " BasePath:" << basePath;
		Core::Log::os("Connection ") << name_ << " " << filename <<" doesn't exists";
		_initializeWeights(nRows, nColumns);
	}
}

void WeightConnection::initializeWeights(const std::string& basePath, const std::string& suffix) {
	require(!isComputing_);
	require(source_->isInitialized());
	require(dest_->isInitialized());

	_initializeWeights(basePath, suffix, source_->nOutputUnits(sourcePort_), dest_->nInputUnits(destPort_));

	require_eq(weights_.nRows(), source_->nOutputUnits(sourcePort_));
	require_eq(weights_.nColumns(), dest_->nInputUnits(destPort_));
}

void WeightConnection::saveWeights(const std::string& basePath, const std::string& suffix) {
	std::string fn = getParamFileName(basePath, suffix);
	Core::Log::os("Connection ") << name_ << ": write weight matrix to " << fn;
	bool areWeightsComputing = weights_.isComputing();
	weights_.finishComputation();
	weights_.write(fn);
	if (areWeightsComputing)
		weights_.initComputation(false);
}

Matrix& WeightConnection::weights() {
	require(hasWeights());
	return weights_;
}

void WeightConnection::_forwardWeightMultiplication(const Matrix& source, Matrix& dest) {
	require_eq(weights_.nRows(), source.nRows());
	require_eq(weights_.nColumns(), dest.nRows());
	require_eq(source.nColumns(), dest.nColumns());
	dest.addMatrixProduct(weights_, source, 1, 1, true, false);
}

void WeightConnection::_backpropagateWeights(const Matrix& source, Matrix& dest) {
	require_eq(dest.nRows(), weights_.nRows());
	require_eq(source.nRows(), weights_.nColumns());
	require_eq(dest.nColumns(), source.nColumns());
	dest.addMatrixProduct(weights_, source, 1, 1, false, false);
}

bool WeightConnection::isTrainable() const {
	return isTrainable_;
}

void WeightConnection::initComputation(bool sync) {
	weights_.initComputation(sync);
	isComputing_ = true;
}

void WeightConnection::finishComputation(bool sync) {
	weights_.finishComputation(sync);
	isComputing_ = false;
}

/*
 * ConvolutionalConnection
 */

const Core::ParameterInt ConvolutionalConnection::paramKernelHeight_("kernel-height", 3, "neural-network.connection");

const Core::ParameterInt ConvolutionalConnection::paramKernelWidth_("kernel-width", 3, "neural-network.connection");

const Core::ParameterInt ConvolutionalConnection::paramDestChannels_("dest-channels", 0, "neural-network.connection");

const Core::ParameterInt ConvolutionalConnection::paramStrideX_("stride-x", 1, "neural-network.connection");

const Core::ParameterInt ConvolutionalConnection::paramStrideY_("stride-y", 1, "neural-network.connection");

ConvolutionalConnection::ConvolutionalConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort,
		u32 destPort, bool isRecurrent, ConnectionType type):
		Precursor(name, source, dest, sourcePort, destPort, isRecurrent, type),
	    kernelHeight_(Core::Configuration::config(paramKernelHeight_, prefix_)),
		kernelWidth_(Core::Configuration::config(paramKernelWidth_, prefix_)),
		destChannels_(Core::Configuration::config(paramDestChannels_, prefix_)),
		strideX_(Core::Configuration::config(paramStrideX_, prefix_)),
		strideY_(Core::Configuration::config(paramStrideY_, prefix_)),
		previousBatchSize_(1),
		cudnnConvolution()
{}

ConvolutionalConnection::~ConvolutionalConnection()
{}

void ConvolutionalConnection::initialize() {
	Precursor::initialize();
	//require(convolution);
	require_gt(kernelHeight_, 0);
	require_gt(kernelWidth_, 0);
	require_gt(destChannels_, 0);
	require_gt(strideX_, 0);
	require_gt(strideY_, 0);
	require_eq(kernelHeight_ % 2, 1);
	require_eq(kernelWidth_ % 2, 1);

	to().setWidth(destPort_, getResultWidth(from().width(sourcePort_), kernelWidth_, strideX_));
	to().setHeight(destPort_, getResultHeight(from().height(sourcePort_), kernelHeight_, strideY_));

	to().setNumberOfChannels(destPort_, destChannels_);
	to().updateNumberOfUnits(destPort_);

#ifdef MODULE_CUDNN
	cudnnConvolution.init(1, from().width(sourcePort_), from().height(sourcePort_), from().nChannels(sourcePort_),
			to().width(destPort_), to().height(destPort_), to().nChannels(destPort_), kernelWidth_/2, kernelHeight_/2,
			strideX_, strideY_, kernelWidth_, kernelHeight_);
#endif
}

void ConvolutionalConnection::initializeWeights(const std::string& basePath, const std::string& suffix) {
	require(!isComputing_);
	require(source_->isInitialized());
	require(dest_->isInitialized());

	_initializeWeights(basePath, suffix, kernelHeight_ * kernelWidth_ * from().nChannels(sourcePort_), destChannels_);

	require_eq(weights_.nRows(), kernelHeight_ * kernelWidth_ * from().nChannels(sourcePort_));
	require_eq(weights_.nColumns(), destChannels_);
}

u32 ConvolutionalConnection::getResultWidth(u32 sourceWidth, u32 kernelWidth, u32 strideX) {
	require_gt(sourceWidth, strideX);
	return ceil((f32)sourceWidth / (f32)strideX);
}
u32 ConvolutionalConnection::getResultHeight(u32 sourceHeight, u32 kernelHeight, u32 strideY) {
	require_gt(sourceHeight, strideY);
	return ceil((f32)sourceHeight / (f32)strideY);
}

void ConvolutionalConnection::backwardPreprocess(const Matrix& source, Matrix& dest) {
	dest.resize(to().nChannels(destPort_), source.nColumns() * getResultWidth(from().width(sourcePort_), kernelWidth_, strideX_) *
			getResultHeight(from().height(sourcePort_), kernelHeight_, strideY_));
	dest.initComputation(false);
	dest.rearrangeBackProp(source, to().nChannels(destPort_));
}

void ConvolutionalConnection::backwardPostprocess(const Matrix& source, Matrix& dest, u32 destColumns) {
	dest.resize(from().width(sourcePort_) * from().height(sourcePort_) * from().nChannels(sourcePort_), destColumns);
	dest.initComputation(false);

	dest.prepareConvolutionSameBackProp(source, from().width(sourcePort_), from().height(sourcePort_),
			from().nChannels(sourcePort_), kernelWidth_, kernelHeight_, strideX_, strideY_);
}

void ConvolutionalConnection::forwardPreprocess(const Matrix& source, Matrix& dest) {

	dest.resize(getResultWidth( from().width(sourcePort_) , kernelWidth_, strideX_) *
			getResultHeight( from().height(sourcePort_) , kernelHeight_, strideY_) * from().nChannels(sourcePort_)
				* kernelWidth_ * kernelHeight_, source.nColumns());

	dest.initComputation(false);
	dest.prepareConvolutionSame(source, from().width(sourcePort_) , from().height(sourcePort_) ,
			from().nChannels(sourcePort_), kernelWidth_, kernelHeight_, strideX_, strideY_);

	dest.reshape(kernelWidth_ * kernelHeight_ * from().nChannels(sourcePort_),
			source.nColumns() * getResultWidth( from().width(sourcePort_) , kernelWidth_, strideX_) *
			getResultHeight( from().height(sourcePort_) , kernelHeight_, strideY_));
}

void ConvolutionalConnection::forward(const Matrix& source, Matrix& dest) {
#ifdef MODULE_CUDNN
	if (source.nColumns() != previousBatchSize_) {
		previousBatchSize_ = source.nColumns();
		cudnnConvolution.updateBatchSize(source.nColumns());
	}
	cudnnConvolution.convolveForward(dest, source, weights_);
#else
	Matrix temp, temp1;
	forwardPreprocess(source, temp);


	temp1.resize(to().nChannels(destPort_), temp.nColumns());
	temp1.initComputation(false);
	temp1.addMatrixProduct(weights_, temp, 0, 1, true, false);

	temp.finishComputation(false);

	forwardPostProcess(temp1, temp, source.nColumns());

	dest.add(temp, 1.0f);
	temp.finishComputation(false);
	temp1.finishComputation(false);
#endif
}

void ConvolutionalConnection::forwardPostProcess(const Matrix& source, Matrix& dest, u32 sourceColumns) {
	dest.resize(destChannels_ * getResultWidth(from().width(sourcePort_), kernelWidth_, strideX_) *
			getResultHeight(from().height(sourcePort_), kernelHeight_, strideY_), sourceColumns);
	dest.initComputation(false);
	dest.rearrange(source, sourceColumns);
}

void ConvolutionalConnection::backward(const Matrix& source, Matrix& dest) {
#ifdef MODULE_CUDNN
	cudnnConvolution.convolveBackwardData(dest, source, weights_);
#else
	Matrix temp, temp1;

	backwardPreprocess(source, temp);

	//convolutional backpropogation
	temp1.resize(kernelWidth_ * kernelHeight_ * from().nChannels(sourcePort_),
			source.nColumns() * getResultWidth(from().width(sourcePort_), kernelWidth_, strideX_) *
			getResultHeight(from().height(sourcePort_), kernelHeight_, strideY_));
	temp1.initComputation(false);
	temp1.addMatrixProduct(weights_, temp, 0, 1, false, false);
	temp1.reshape( kernelHeight_ * kernelWidth_ * from().nChannels(sourcePort_) *
			getResultWidth(from().width(sourcePort_), kernelWidth_, strideX_) *
			getResultHeight(from().height(sourcePort_), kernelHeight_, strideY_) ,dest.nColumns());

	temp.finishComputation(false);
	backwardPostprocess(temp1, temp, dest.nColumns());

	dest.add(temp, 1.0f);
	temp.finishComputation(false);
	temp1.finishComputation(false);
#endif
}

void ConvolutionalConnection::_forwardWeightMultiplication(const Matrix& source, Matrix& dest) {
	forward(source, dest);
}

void ConvolutionalConnection::_backpropagateWeights(const Matrix & source, Matrix &dest) {
	backward(source, dest);
}
void ConvolutionalConnection::backwardWRTKernel(Matrix &weightsGradient, const Matrix &activationIn, const Matrix &errorSignalOut) {
#ifdef MODULE_CUDNN
	cudnnConvolution.convolveBackwardFilter(weightsGradient, activationIn, errorSignalOut);
#endif
}
void ConvolutionalConnection::initComputation(bool sync) {
	Precursor::initComputation(sync);
	isComputing_ = true;
}

void ConvolutionalConnection::finishComputation(bool sync) {
	Precursor::finishComputation(sync);
	isComputing_ = false;
}

u32 ConvolutionalConnection::kernelWidth() {
	return kernelWidth_;
}

u32 ConvolutionalConnection::kernelHeight() {
	return kernelHeight_;
}

u32 ConvolutionalConnection::destChannels() {
	return destChannels_;
}
/*
 * Valid Convolutional Connection i.e no zero padding
 */
ValidConvolutionalConnection::ValidConvolutionalConnection(const char *name, BaseLayer* source,
		BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, enum ConnectionType type):
				Precursor(name, source, dest, sourcePort, destPort, isRecurrent, type)
{
}
ValidConvolutionalConnection::~ValidConvolutionalConnection() {}

u32 ValidConvolutionalConnection::getResultHeight(u32 sourceHeight, u32 kernelHeight, u32 strideY) {
	require_gt(sourceHeight, kernelHeight);
	require_gt(sourceHeight - kernelHeight + 1, strideY);
	return ceil((f32)(sourceHeight - kernelHeight + 1) / (f32)strideY);
}
u32 ValidConvolutionalConnection::getResultWidth(u32 sourceWidth, u32 kernelWidth, u32 strideX) {
	require_gt(sourceWidth, kernelWidth);
	require_gt(sourceWidth - kernelWidth + 1, strideX);
	return ceil((f32)(sourceWidth - kernelWidth + 1) / (f32)strideX);
}
void ValidConvolutionalConnection::forwardPreprocess(const Matrix& source, Matrix& dest) {
	dest.resize(getResultWidth( from().width(sourcePort_) , kernelWidth_, strideX_) *
			getResultHeight( from().height(sourcePort_) , kernelHeight_, strideY_) * from().nChannels(sourcePort_)
				* kernelWidth_ * kernelHeight_, source.nColumns());

	dest.initComputation(false);
	dest.prepareConvolution(source, from().width(sourcePort_) , from().height(sourcePort_) , from().nChannels(sourcePort_), kernelWidth_, kernelHeight_, strideX_, strideY_);
	dest.reshape(kernelWidth_ * kernelHeight_ * from().nChannels(sourcePort_),
			source.nColumns() * getResultWidth( from().width(sourcePort_) , kernelWidth_, strideX_) *
			getResultHeight( from().height(sourcePort_) , kernelHeight_, strideY_));
}
void ValidConvolutionalConnection::backwardPostprocess(const Matrix& source, Matrix& dest, u32 destColumns) {
	dest.resize(from().width(sourcePort_) * from().height(sourcePort_) * from().nChannels(sourcePort_), destColumns);
	dest.initComputation(false);

	dest.prepareConvolutionBackProp(source, from().width(sourcePort_), from().height(sourcePort_), from().nChannels(sourcePort_), kernelWidth_, kernelHeight_);
}
void ValidConvolutionalConnection::_forwardWeightMultiplication(const Matrix &source, Matrix &dest) {
	forward(source, dest);
}
void ValidConvolutionalConnection::_backpropagateWeights(const Matrix &source, Matrix &dest) {
	backward(source, dest);
}
