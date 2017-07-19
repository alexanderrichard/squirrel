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
 * ActivationLayer.cc
 *
 *  Created on: May 14, 2014
 *      Author: richard
 */

#include "ActivationLayer.hh"

using namespace Nn;

/*
 * SigmoidLayer
 */
SigmoidLayer::SigmoidLayer(const char* name) :
		Precursor(name)
{}

void SigmoidLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).sigmoid();
}

void SigmoidLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithSigmoidDerivative(activationsOut(timeframe, port));
}

/*
 * TanhLayer
 */
TanhLayer::TanhLayer(const char* name) :
		Precursor(name)
{}

void TanhLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).tanh();
}

void TanhLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithTanhDerivative(activationsOut(timeframe, port));
}

/*
 * SoftmaxLayer
 */
SoftmaxLayer::SoftmaxLayer(const char* name) :
		Precursor(name)
{}

void SoftmaxLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).softmax();
}

void SoftmaxLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).multiplicationWithSoftmaxDerivative(activationsOut(timeframe, port));
}

/*
 * MaxLayer
 */
MaxLayer::MaxLayer(const char* name) :
		Precursor(name)
{}

void MaxLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).max();
}

void MaxLayer::backpropagate(u32 timeframe, u32 port) {
	std::cerr << "Nn::MaxLayer::backpropagate: Layer " << name_ << " is not differentiable. Abort." << std::endl;
	exit(1);
}

/*
 * RectifiedLayer
 */
RectifiedLayer::RectifiedLayer(const char* name) :
		Precursor(name)
{}

void RectifiedLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).ensureMinimalValue(0);
}

void RectifiedLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithClippedDerivative(activationsOut(timeframe, port), 0.0, Types::inf<Float>());
}

/*
 * ClippedLayer
 */
const Core::ParameterFloat ClippedLayer::paramThresholdLeft_("left-threshold", 0.0, "neural-network.layer");

const Core::ParameterFloat ClippedLayer::paramThresholdRight_("right-threshold", 1.0, "neural-network.layer");

ClippedLayer::ClippedLayer(const char* name) :
		Precursor(name),
		thresholdLeft_(Core::Configuration::config(paramThresholdLeft_, prefix_)),
		thresholdRight_(Core::Configuration::config(paramThresholdRight_, prefix_))
{
	require_lt(thresholdLeft_, thresholdRight_);
}

void ClippedLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).ensureMinimalValue(thresholdLeft_);
	activationsIn(t, port).ensureMaximalValue(thresholdRight_);
}

void ClippedLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithClippedDerivative(activationsOut(timeframe, port), thresholdLeft_, thresholdRight_);
}

/*
 * ExponentialLayer
 */
ExponentialLayer::ExponentialLayer(const char* name) :
		Precursor(name)
{}

void ExponentialLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).exp();
}

void ExponentialLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplication(activationsOut(timeframe, port));
}

/*
 * LogarithmicLayer
 */
LogarithmicLayer::LogarithmicLayer(const char* name) :
		Precursor(name)
{}

void LogarithmicLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).ensureMinimalValue(0);
	activationsIn(t, port).log();
}

void LogarithmicLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithLogDerivative(activationsOut(timeframe, port));
}

/*
 * L2NormalizationLayer
 */
L2NormalizationLayer::L2NormalizationLayer(const char* name) :
		Precursor(name)
{}

void L2NormalizationLayer::addTimeframe(u32 minibatchSize) {
	Precursor::addTimeframe(minibatchSize);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).addTimeframe(1, minibatchSize);
}

void L2NormalizationLayer::addEmptyTimeframe() {
	Precursor::addEmptyTimeframe();
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).addTimeframe(0, 0);
}

void L2NormalizationLayer::setActivationVisibility(u32 timeframe, u32 nVisibleColumns) {
	Precursor::setActivationVisibility(timeframe, nVisibleColumns);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).at(timeframe).setVisibleColumns(nVisibleColumns);
}

void L2NormalizationLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	normalization_.resize(nOutputPorts());
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).setMaximalMemory(maxMemory);
}

void L2NormalizationLayer::resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns) {
	Precursor::resizeTimeframe(timeframe, nRows, nColumns);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).at(timeframe).resize(1, nColumns);
}

void L2NormalizationLayer::setMaximalMemory(u32 maxMemory) {
	Precursor::setMaximalMemory(maxMemory);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).setMaximalMemory(maxMemory);
}

void L2NormalizationLayer::reset() {
	Precursor::reset();
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).reset();
}

void L2NormalizationLayer::forward(u32 port) {
	require_lt(port, nOutputPorts());
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	Vector tmp;
	tmp.initComputation();
	tmp.swap(normalization_.at(port).getLast());
	tmp.columnwiseInnerProduct(activationsIn(t, port), activationsIn(t, port));
	tmp.signedPow(0.5);
	tmp.ensureMinimalValue(Types::absMin<Float>()); // prevent division by 0
	activationsIn(t, port).divideColumnsByScalars(tmp);
	normalization_.at(port).getLast().swap(tmp);
	normalization_.at(port).getLast().safeResize(1, normalization_.at(port).getLast().nRows());
}

void L2NormalizationLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0) {
		Vector tmp;
		tmp.initComputation();
		tmp.swap(normalization_.at(port).at(timeframe));
		errorSignalOut(timeframe, port).multiplicationWithL2NormalizationDerivative(activationsOut(timeframe, port), tmp);
		normalization_.at(port).at(timeframe).swap(tmp);
		normalization_.at(port).getLast().safeResize(1, normalization_.at(port).getLast().nRows());
	}
}

void L2NormalizationLayer::initComputation(bool sync) {
	if (isComputing_)
		return;
	Precursor::initComputation(sync);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).initComputation(sync);
}

void L2NormalizationLayer::finishComputation(bool sync) {
	if (!isComputing_)
		return;
	Precursor::finishComputation(sync);
	for (u32 port = 0; port < nOutputPorts(); port++)
		normalization_.at(port).finishComputation(sync);
}

/*
 * PowerNormalizationLayer
 */
const Core::ParameterFloat PowerNormalizationLayer::paramAlpha_("power", 0.5, "neural-network.layer");

PowerNormalizationLayer::PowerNormalizationLayer(const char* name) :
		Precursor(name),
		alpha_(Core::Configuration::config(paramAlpha_, prefix_))
{}

void PowerNormalizationLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsIn(t, port).signedPow(alpha_);
}

void PowerNormalizationLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0) {
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithSignedPowDerivative(activationsOut(timeframe, port), alpha_);
	}
}

/*
 * SequenceNormalizationLayer
 */
SequenceLengthNormalizationLayer::SequenceLengthNormalizationLayer(const char* name) :
		Precursor(name)
{}

void SequenceLengthNormalizationLayer::forward(u32 port) {
	Precursor::forward(port);
	// sequences must have the same length for each port -> accumulate sequence length based on port 0
	if (port == 0) {
		u32SequenceLengths_.resize(latestActivations(port).nColumns(), 0);
		for (u32 i = 0; i < latestActivations(port).nColumns(); i++) {
			u32SequenceLengths_.at(i)++;
		}
	}
}

void SequenceLengthNormalizationLayer::finalizeForwarding() {
	Precursor::finalizeForwarding();
	sequenceLengths_.finishComputation(false);
	sequenceLengths_.resize(u32SequenceLengths_.size());
	for (u32 i = 0; i < sequenceLengths_.size(); i++) {
		sequenceLengths_.at(i) = (Float)u32SequenceLengths_.at(i);
	}
	sequenceLengths_.initComputation();
	for (u32 port = 0; port < nOutputPorts(); port ++)
		latestActivations(port).divideColumnsByScalars(sequenceLengths_);
	// reset u32SequenceLengths_ for next mini-batch
	u32SequenceLengths_.clear();
}

void SequenceLengthNormalizationLayer::backpropagate(u32 timeframe, u32 port) {
	require(sequenceLengths_.isComputing());
	require_le(sequenceLengths_.nRows(), latestActivations(port).nColumns());
	Precursor::backpropagate(timeframe, port);
	// if sequence end: normalize by sequence length
	if (timeframe == nTimeframes() - 1) {
		require_eq(errorSignalOut(timeframe, port).nColumns(), sequenceLengths_.nRows());
		errorSignalOut(timeframe, port).divideColumnsByScalars(sequenceLengths_);
	}
}

/*
 * TemporalReversionLayer
 */
TemporalReversionLayer::TemporalReversionLayer(const char* name) :
		Precursor(name)
{
	useBias_ = false;
}

void TemporalReversionLayer::finalizeForwarding() {
	Precursor::finalizeForwarding();
	for (u32 port = 0; port < nOutputPorts(); port++)
		activations_.at(port).revertTemporalOrder();
}

void TemporalReversionLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// temporal reversion of error signals when all error signals have been computed
	if ((timeframe == 0) && (errorSignals_.at(port).nTimeframes() > 1)) {
		errorSignals_.at(port).at(timeframe+1).setVisibleColumns(activations_.at(port).at(timeframe+1).nColumns());
		errorSignals_.at(port).revertTemporalOrder();
		errorSignals_.at(port).at(timeframe+1).setVisibleColumns(activations_.at(port).at(timeframe).nColumns());
	}
}
