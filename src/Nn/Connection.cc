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

#include "Connection.hh"
#include "Layer.hh"
#include "MatrixContainer.hh"
#include <Math/Random.hh>
#include <sstream>

using namespace Nn;

/*
 * Connection
 */

const Core::ParameterEnum Connection::paramConnectionType_("type",
		"plain-connection, weight-connection",
		"weight-connection", "neural-network.connection");

const Core::ParameterFloat Connection::paramWeightScale_("weight-scale", 1.0, "neural-network.connection");

Connection::Connection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, ConnectionType type) :
		name_(name),
		prefix_(std::string("neural-network.").append(name)),
		source_(source),
		dest_(dest),
		sourcePort_(sourcePort),
		destPort_(destPort),
		isComputing_(false),
		connectionType_(type),
		weightScale_(Core::Configuration::config(paramWeightScale_, prefix_))
{}

void Connection::initialize() {
	if (source_->useDropout()) {
		weightScale_ /= 1 - source_->dropoutProbability();
	}
}

bool Connection::isRecurrent() const {
	return ((source_ == dest_) && (dest_ != 0));
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
	dest.add(source, weightScale_);
}

void Connection::forwardWeightMultiplication() {
	require(isComputing_);
	require(source_);
	require(dest_);
	require_eq(source_->nTimeframes(), dest_->nTimeframes());
	u32 t = source_->nTimeframes() - 1; // the latest time frame index
	// note: dest_->activations(t) is not implicitly reset
	if (isRecurrent() && (dest_->nTimeframes() > 1)) {
		_forwardWeightMultiplication(source_->activationsOut(t-1, sourcePort_), dest_->activationsIn(t, destPort_));
	}
	else if (!isRecurrent()) {
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
 * WeightConnection
 */
const Core::ParameterBool WeightConnection::paramIsTrainable_("is-trainable", true, "neural-network.connection");

const Core::ParameterEnum WeightConnection::paramWeightInitialization_("weight-initialization", "random, zero, identity",
		"random", "neural-network.connection");

const Core::ParameterFloat WeightConnection::paramRandomWeightMin_("random-weight-min", -0.1, "neural-network.connection");

const Core::ParameterFloat WeightConnection::paramRandomWeightMax_("random-weight-max", 0.1, "neural-network.connection");

const Core::ParameterString WeightConnection::paramOldWeightsFilename_("load-weights-from", "", "neural-network.connection");

const Core::ParameterString WeightConnection::paramNewWeightsFilename_("write-weights-to", "", "neural-network.connection");

WeightConnection::WeightConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, ConnectionType type) :
		Precursor(name, source, dest, sourcePort, destPort, type),
		oldWeightsFile_(Core::Configuration::config(paramOldWeightsFilename_, prefix_)),
		newWeightsFile_(Core::Configuration::config(paramNewWeightsFilename_, prefix_)),
		isTrainable_(Core::Configuration::config(paramIsTrainable_, prefix_))
{}

void WeightConnection::initialize() {
	Precursor::initialize();
	require(source_->isInitialized());
	require(dest_->isInitialized());
	initializeWeights();
}

void WeightConnection::_initializeWeights(u32 nRows, u32 nColumns) {
	require(!isComputing_);
	weights_.resize(nRows, nColumns);
	Float min = Core::Configuration::config(paramRandomWeightMin_, prefix_);
	Float max = Core::Configuration::config(paramRandomWeightMax_, prefix_);
	require_lt(min, max);
	switch (Core::Configuration::config(paramWeightInitialization_, prefix_)) {
	case zero:
		weights_.setToZero();
		break;
	case identity:
		weights_.setToZero();
		for (u32 i = 0; i < std::min(weights_.nRows(), weights_.nColumns()); i++)
			weights_.at(i,i) = 1;
		break;
	case random:
		Math::RandomNumberGenerator rand;
		for (u32 i = 0; i < weights_.nRows(); i++) {
			for (u32 j = 0; j < weights_.nColumns(); j++) {
				weights_.at(i,j) = rand.random(min, max);
			}
		}
		break;
	default: // cannot happen
		break;
	}
}

void WeightConnection::initializeWeights() {
	require(!isComputing_);
	if (oldWeightsFile_.empty()) {
		std::string initializationMethod;
		switch (Core::Configuration::config(paramWeightInitialization_, prefix_)) {
		case zero:
			initializationMethod = "zero";
			break;
		case identity:
			initializationMethod = "identity";
			break;
		case random:
		default:
			initializationMethod = "random";
			break;
		}
		Core::Log::os("Connection ") << name_
				<< ": no file to load weights from. Use initialization: "
				<< initializationMethod;
		_initializeWeights(source_->nUnits(), dest_->nUnits());
	}
	else {
		Core::Utils::appendSuffix(oldWeightsFile_, weightsFileSuffix_);
		Core::Log::os("Connection ") << name_ << ": read weight matrix from " << oldWeightsFile_;
		weights_.read(oldWeightsFile_);
		require_eq(weights_.nRows(), source_->nUnits());
		require_eq(weights_.nColumns(), dest_->nUnits());
	}
	// in case of dropout: scale with dropout probability (if no dropout, weight scale is 1.0)
	for (u32 i = 0; i < weights_.nRows(); i++) {
		for (u32 j = 0; j < weights_.nColumns(); j++) {
			weights_.at(i, j) *= weightScale_;
		}
	}
}

void WeightConnection::setOldWeightsFile(const std::string& filename) {
	oldWeightsFile_ = filename;
}

void WeightConnection::setNewWeightsFile(const std::string& filename) {
	newWeightsFile_ = filename;
}

void WeightConnection::saveWeights(const std::string& suffix) {
	if (newWeightsFile_.empty()) {
		Core::Log::os("Connection ") << name_ << ": no file to write weights to specified. Do not save weights.";
	}
	else {
		std::string fn = newWeightsFile_;
		Core::Utils::appendSuffix(fn, suffix);
		Core::Utils::appendSuffix(fn, weightsFileSuffix_);
		Core::Log::os("Connection ") << name_ << ": write weight matrix to " << fn;
		bool areWeightsComputing = weights_.isComputing();
		weights_.finishComputation();
		// if dropout is used, rescale weights
		if (weightScale_ != 1.0) {
			for (u32 i = 0; i < weights_.nRows(); i++) {
				for (u32 j = 0; j < weights_.nColumns(); j++) {
					weights_.at(i, j) /= weightScale_;
				}
			}
		}
		weights_.write(fn);
		// undo the rescaling
		if (weightScale_ != 1.0) {
			for (u32 i = 0; i < weights_.nRows(); i++) {
				for (u32 j = 0; j < weights_.nColumns(); j++) {
					weights_.at(i, j) *= weightScale_;
				}
			}
		}
		if (areWeightsComputing)
			weights_.initComputation(false);
	}
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
 * Connection factory
 */
Connection* Connection::createConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort) {
	Connection* connection = 0;
	std::string prefix("neural-network.");
	prefix.append(name);
	ConnectionType type = (ConnectionType) Core::Configuration::config(paramConnectionType_, prefix);
	switch (type) {
	case weightConnection:
		connection = new WeightConnection(name, source, dest, sourcePort, destPort, type);
		break;
	case plainConnection:
	default:
		connection = new Connection(name, source, dest, sourcePort, destPort, type);
		break;
	}
	return connection;
}



