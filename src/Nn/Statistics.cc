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
 * Statistics.cc
 *
 *  Created on: May 20, 2014
 *      Author: richard
 */

#include "Statistics.hh"
#include "Connection.hh"
#include <sstream>

using namespace Nn;

Statistics::Statistics(StatisticTypes types) :
		needsClassificationStatistics_((types & classificationStatistics) != 0 ? true : false),
		needsObjectiveFunction_((types & objectiveFunctionStatistics) != 0 ? true : false),
		needsGradient_((types & gradientStatistics) != 0 ? true : false),
		needsSequenceCount_((types & sequenceCount) != 0 ? true : false),
		isNormalized_(false),
		isInitialized_(false),
		nObservations_(0),
		nSequences_(0),
		nClassificationErrors_(0),
		objectiveFunctionValue_(0),
		isComputing_(false)
{}

void Statistics::initialize(const NeuralNetwork& network) {
	if (needsGradient_) {
		// initialize bias gradient
		u32 i = 0;
		for (u32 l = 0; l < network.nLayer(); l++) {
			if (network.layer(l).isBiasTrainable()) {
				for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
					std::stringstream s;
					s << network.layer(l).name() << ".port-" << port;
					layerNameToIndex_[s.str()] = i;
					biasGradient_.push_back(Vector(network.layer(l).bias(port).nRows()));
					i++;
				}
			}
		}
		// initialize weights gradient
		i = 0;
		for (u32 c = 0; c < network.nConnections(); c++) {
			if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
				connectionNameToIndex_[network.connection(c).name()] = i;
				WeightConnection* conn = dynamic_cast<WeightConnection*>(&(network.connection(c)));
				weightsGradient_.push_back(Matrix(conn->weights().nRows(), conn->weights().nColumns()));
				i++;
			}
		}
	}

	// reset everything
	reset();
	isInitialized_ = true;
}

void Statistics::reset() {
	isNormalized_ = false;
	nObservations_ = 0;
	if (needsSequenceCount_) {
		nSequences_ = 0;
	}
	if (needsClassificationStatistics_) {
		nClassificationErrors_ = 0;
	}
	if (needsObjectiveFunction_) {
		objectiveFunctionValue_ = 0;
	}
	if (needsGradient_) {
		// reset bias gradient
		for (u32 i = 0; i < biasGradient_.size(); i++) {
			biasGradient_.at(i).setToZero();
		}
		// reset weights gradient
		for (u32 i = 0; i < weightsGradient_.size(); i++) {
			weightsGradient_.at(i).setToZero();
		}
	}
}

void Statistics::saveStatistics(const std::string& filename) {
	require(!filename.empty());
	require(!isComputing_);
	Core::BinaryStream o(filename.c_str(), std::ios::out);
	/* cosistency values */
	u8 version = 1;
	o << version;
	// number of bias gradients
	o << (u32)biasGradient_.size();
	// dimensions of each bias gradient vector
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		o << biasGradient_.at(i).nRows();
	}
	// number of weight gradients
	o << (u32)weightsGradient_.size();
	// dimensions of each weight gradient matrix
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		o << weightsGradient_.at(i).nRows() << weightsGradient_.at(i).nColumns();
	}
	/* actual statistics */
	// the actual bias gradient values
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		for (u32 j = 0; j < biasGradient_.at(i).nRows(); j++) {
			o << biasGradient_.at(i).at(j);
		}
	}
	// the actual weight gradient values
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		for (u32 row = 0; row < weightsGradient_.at(i).nRows(); row++) {
			for (u32 column = 0; column < weightsGradient_.at(i).nColumns(); column++) {
				o << weightsGradient_.at(i).at(row, column);
			}
		}
	}
	// base statistics
	o << needsClassificationStatistics_;
	o << needsObjectiveFunction_;
	o << needsGradient_;
	o << isNormalized_;
	o << nObservations_;
	o << nClassificationErrors_;
	o << objectiveFunctionValue_;
	o.close();
}

bool Statistics::checkConsistency(Core::BinaryStream& s) {
	require(isInitialized_);
	u8 version; s >> version;
	if (version != 1) { return false; }
	u32 size; s >> size;
	if (size != biasGradient_.size()) { return false; }
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		s >> size;
		if (size != biasGradient_.at(i).nRows()) { return false; }
	}
	s >> size;
	if (size != weightsGradient_.size()) { return false; }
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		s >> size;
		if (size != weightsGradient_.at(i).nRows()) { return false; }
		s >> size;
		if (size != weightsGradient_.at(i).nColumns()) { return false; }
	}
	return true;
}

void Statistics::loadStatistics(const std::string& filename) {
	require(!filename.empty());
	require(!isComputing_);
	require(isInitialized_);
	Core::Log::os("Load statistics from ") << filename;
	Core::BinaryStream s(filename.c_str(), std::ios::in);
	if (!checkConsistency(s)) {
		std::cerr << "Nn::Statistics::loadStatistics: statistics file " << filename << " is not consistent with the network. Abort." << std::endl;
		exit(1);
	}
	// read bias gradient
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		for (u32 j = 0; j < biasGradient_.at(i).nRows(); j++) {
			s >> biasGradient_.at(i).at(j);
		}
	}
	// read weights gradient
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		for (u32 row = 0; row < weightsGradient_.at(i).nRows(); row++) {
			for (u32 column = 0; column < weightsGradient_.at(i).nColumns(); column++) {
				s >> weightsGradient_.at(i).at(row, column);
			}
		}
	}
	// base statistics
	s >> needsClassificationStatistics_;
	s >> needsObjectiveFunction_;
	s >> needsGradient_;
	s >> isNormalized_;
	s >> nObservations_;
	s >> nClassificationErrors_;
	s >> objectiveFunctionValue_;
	s.close();
}

void Statistics::normalize() {
	require(!isNormalized_);
	if (needsObjectiveFunction_) {
		objectiveFunctionValue_ /= nObservations_;
	}
	if (needsGradient_) {
		for (u32 i = 0; i < biasGradient_.size(); i++)
			biasGradient_.at(i).scale(1.0 / nObservations_);
		for (u32 i = 0; i < weightsGradient_.size(); i++)
			weightsGradient_.at(i).scale(1.0 / nObservations_);
	}
	isNormalized_ = true;
}

u32 Statistics::nObservations() {
	return nObservations_;
}

u32 Statistics::nSequences() {
	require(needsSequenceCount_);
	return nSequences_;
}

u32 Statistics::nClassificationErrors() {
	require(needsClassificationStatistics_);
	return nClassificationErrors_;
}

Float Statistics::objectiveFunction() {
	require(needsObjectiveFunction_);
	return objectiveFunctionValue_;
}

Vector& Statistics::biasGradient(const std::string& layerName, u32 port) {
	require(needsGradient_);
	std::stringstream s;
	s << layerName << ".port-" << port;
	require(layerNameToIndex_.find(s.str()) != layerNameToIndex_.end());
	return biasGradient_.at(layerNameToIndex_[s.str()]);
}

Matrix& Statistics::weightsGradient(const std::string& connectionName) {
	require(needsGradient_);
	require(connectionNameToIndex_.find(connectionName) != connectionNameToIndex_.end());
	return weightsGradient_.at(connectionNameToIndex_[connectionName]);
}

Float Statistics::gradientNorm() {
	require(needsGradient_);
	Float nrm = 0.0;
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		nrm += biasGradient_.at(i).asum();
	}
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		nrm += weightsGradient_.at(i).l1norm();
	}
	return nrm;
}

void Statistics::addToObjectiveFunction(Float value) {
	require(needsObjectiveFunction_);
	objectiveFunctionValue_ += value;
}

void Statistics::increaseNumberOfObservations(u32 nObservations) {
	nObservations_ += nObservations;
}

void Statistics::increaseNumberOfSequences(u32 nSequences) {
	nSequences_ += nSequences;
}

void Statistics::increaseNumberOfClassificationErrors(u32 nErrors) {
	require(needsClassificationStatistics_);
	nClassificationErrors_ += nErrors;
}

void Statistics::initComputation(bool sync) {
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		biasGradient_.at(i).initComputation(sync);
	}
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		weightsGradient_.at(i).initComputation(sync);
	}
	isComputing_ = true;
}

void Statistics::finishComputation(bool sync) {
	for (u32 i = 0; i < biasGradient_.size(); i++) {
		biasGradient_.at(i).finishComputation(sync);
	}
	for (u32 i = 0; i < weightsGradient_.size(); i++) {
		weightsGradient_.at(i).finishComputation(sync);
	}
	isComputing_ = false;
}
