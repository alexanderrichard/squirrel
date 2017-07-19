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

 * MultiPortLayer.cc
 *
 *  Created on: May 17, 2016
 *      Author: ahsan
 */
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "MultiPortLayer.hh"
#include <Math/CudaMatrix.hh>
#include "MatrixContainer.hh"
#include "NeuralNetwork.hh"
#include <sys/stat.h>

using namespace Nn;

MultiPortLayer::MultiPortLayer(const char* name) :
		Precursor(name),
		destWidth_(width_),
		destHeight_(height_),
		nOutUnits_(nUnits_)
{}

void MultiPortLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	outputActivations_.resize(nOutputPorts());
	outputErrorSignals_.resize(nOutputPorts());
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).setMaximalMemory(maxMemory);
		if(trainingMode_)
			outputErrorSignals_.at(i).setMaximalMemory(maxMemory);
	}
}

Matrix& MultiPortLayer::activationsOut(u32 timeframe, u32 port) {
	require_lt(port, nOutputPorts());
	require_lt(timeframe, nTimeframes());
	return outputActivations_.at(port).at(timeframe);
}

Matrix& MultiPortLayer::errorSignalOut(u32 timeframe, u32 port) {
	require(trainingMode_);
	require_lt(port, nOutputPorts());
	require_lt(timeframe, nTimeframes());
	return outputErrorSignals_.at(port).at(timeframe);
}

void MultiPortLayer::addTimeframe(u32 minibatchSize) {
	Precursor::addTimeframe(minibatchSize);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).addTimeframe(nOutputUnits(i), minibatchSize);
		outputActivations_.at(i).getLast().setToZero();
		if(trainingMode_) {
			outputErrorSignals_.at(i).addTimeframe(nOutputUnits(i), minibatchSize);
			outputErrorSignals_.at(i).getLast().setToZero();
		}
	}
}

void MultiPortLayer::addEmptyTimeframe() {
	Precursor::addEmptyTimeframe();
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).addTimeframe(0, 0);
		if(trainingMode_)
			outputErrorSignals_.at(i).addTimeframe(0, 0);
	}
}

void MultiPortLayer::setMaximalMemory(u32 maxMemory) {
	Precursor::setMaximalMemory(maxMemory);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).setMaximalMemory(maxMemory);
		if(trainingMode_)
			outputErrorSignals_.at(i).setMaximalMemory(maxMemory);
	}
}

void MultiPortLayer::resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns) {
	Precursor::resizeTimeframe(timeframe, nRows, nColumns);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).at(timeframe).resize(nRows, nColumns);
		if (trainingMode_)
			outputErrorSignals_.at(i).at(timeframe).resize(nRows, nColumns);
	}
}

void MultiPortLayer::reset() {
	Precursor::reset();
	for (u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).reset();
		if (trainingMode_)
			outputErrorSignals_.at(i).reset();
	}
}

void MultiPortLayer::setActivationVisibility(u32 timeframe, u32 nVisibleColumns) {
	Precursor::setActivationVisibility(timeframe, nVisibleColumns);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).at(timeframe).setVisibleColumns(nVisibleColumns);
	}
}

void MultiPortLayer::setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns) {
	Precursor::setErrorSignalVisibility(timeframe, nVisibleColumns);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputErrorSignals_.at(i).at(timeframe).setVisibleColumns(nVisibleColumns);
	}
}

void MultiPortLayer::initComputation(bool sync) {
	if (isComputing_)
		return;
	Precursor::initComputation(sync);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).initComputation(sync);
		if(trainingMode_)
			outputErrorSignals_.at(i).initComputation(sync);
	}
}

void MultiPortLayer::finishComputation(bool sync) {
	if (!isComputing_)
		return;
	Precursor::finishComputation(sync);
	for(u32 i = 0; i < nOutputPorts(); i++) {
		outputActivations_.at(i).finishComputation(sync);
		if(trainingMode_)
			outputErrorSignals_.at(i).finishComputation(sync);
	}
}

void MultiPortLayer::setWidth(u32 port, u32 width) {
	Precursor::setWidth(port, width);
	destWidth_ = width;
}

void MultiPortLayer::setHeight(u32 port, u32 height) {
	Precursor::setHeight(port, height);
	destHeight_ = height;
}

void MultiPortLayer::updateNumberOfUnits(u32 port) {
	Precursor::updateNumberOfUnits(port);
	nOutUnits_ = nUnits_;
}


/*
 * TriangleActivationLayer
 */
TriangleActivationLayer::TriangleActivationLayer(const char* name) :
		Precursor(name)
{}

void TriangleActivationLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsOut(t, port).copy(activationsIn(t, port));
	activationsOut(t, port).triangle();
}

void TriangleActivationLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	// error signal is zero if no outgoing layers exist, so the multiplication would not be necessary
	if (nOutgoingConnections(port) > 0)
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithTriangleDerivative(activationsIn(timeframe, port));
}


/*
 * TemporalAveragingLayer + relu
 */
TemporalAveragingLayer::TemporalAveragingLayer(const char* name) :
		Precursor(name)
{}

void TemporalAveragingLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
	activationsOut(t, port).add(activationsIn(t, port), (Float)1.0 / (t+1));
	if (t > 0) {
		activationsOut(t, port).safeResize(activationsIn(t-1, port).nRows(), activationsIn(t-1, port).nColumns());
		activationsOut(t, port).add(activationsIn(t-1, port), (Float)t / (t+1));
		activationsOut(t, port).safeResize(activationsIn(t, port).nRows(), activationsIn(t, port).nColumns());
	}
	// apply relu
	activationsOut(t, port).ensureMinimalValue(0.0);
}

void TemporalAveragingLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	errorSignalOut(timeframe, port).elementwiseMultiplicationWithClippedDerivative(activationsOut(timeframe, port), 0.0, Types::inf<Float>());
	errorSignalIn(timeframe, port).add(errorSignalOut(timeframe, port), (Float)1.0 / (timeframe + 1));
	if (timeframe > 0) {
		errorSignalOut(timeframe, port).safeResize(errorSignalIn(timeframe-1, port).nRows(), errorSignalIn(timeframe-1, port).nColumns());
		errorSignalIn(timeframe-1, port).add(errorSignalOut(timeframe, port), (Float)(timeframe) / (timeframe + 1));
		errorSignalOut(timeframe, port).safeResize(activationsOut(timeframe, port).nRows(), activationsOut(timeframe, port).nColumns());
	}
}


/*
 * MultiplicationLayer
 */
MultiplicationLayer::MultiplicationLayer(const char* name) :
		Precursor(name)
{}

void MultiplicationLayer::forward() {
	Precursor::forward();
	u32 t = nTimeframes() - 1;
	activationsOut(t, 0).copy(activationsIn(t, 0));
	for (u32 port = 1; port < nInputPorts(); port++)
		activationsOut(t, 0).elementwiseMultiplication(activationsIn(t, port));
}

void MultiplicationLayer::backpropagate(u32 timeframe) {
	Precursor::backpropagate(timeframe);
	// error signal at each input port: multiplication with activations of all other input ports
	for (u32 port = 0; port < nInputPorts(); port++) {
		errorSignalIn(timeframe, port).copy(errorSignalOut(timeframe, 0));
		for (u32 p = 0; p < nInputPorts(); p++) {
			if (p != port)
				errorSignalIn(timeframe, port).elementwiseMultiplication(activationsIn(timeframe, p));
		}
	}
}


/*
 * ConcatenationLayer
 */
ConcatenationLayer::ConcatenationLayer(const char* name) :
		Precursor(name)
{}

void ConcatenationLayer::updateNumberOfUnits(u32 port) {
	Precursor::updateNumberOfUnits(port);
	nOutUnits_ = nOutputUnits(port);
}

u32 ConcatenationLayer::nOutputUnits(u32 port) const {
	u32 res = 0;
	for (u32 port = 0; port < nInputPorts(); port++)
		res += nInputUnits(port);
	return res;
}

void ConcatenationLayer::forward() {
	Precursor::forward();
	u32 t = nTimeframes() - 1;
	u32 row = 0;
	for (u32 port = 0; port < nInputPorts(); port++) {
		activationsOut(t, 0).copyBlockFromMatrix(activationsIn(t, port), 0, 0, row, 0,
				activationsIn(t, port).nRows(), activationsIn(t, port).nColumns());
		row += activationsIn(t, port).nRows();
	}
}

void ConcatenationLayer::backpropagate(u32 timeframe) {
	Precursor::backpropagate(timeframe);
	u32 row = 0;
	for (u32 port = 0; port < nInputPorts(); port++) {
		errorSignalIn(timeframe, port).copyBlockFromMatrix(errorSignalOut(timeframe, 0), row, 0, 0, 0,
				errorSignalIn(timeframe, port).nRows(), errorSignalIn(timeframe, port).nColumns());
		row += errorSignalIn(timeframe, port).nRows();
	}
}

/*
 * SpatialPoolingLayer
 */
const Core::ParameterInt SpatialPoolingLayer::paramGridSize_("grid-size", 2, "neural-network.layer");

const Core::ParameterInt SpatialPoolingLayer::paramStride_("stride", 2, "neural-network.layer");

const Core::ParameterInt SpatialPoolingLayer::paramPadX_("pad-x", -1, "neural-network.layer");

const Core::ParameterInt SpatialPoolingLayer::paramPadY_("pad-y", -1, "neural-network.layer");

SpatialPoolingLayer::SpatialPoolingLayer(const char* name):
		Precursor(name),
		gridSize_(Core::Configuration::config(paramGridSize_, prefix_)),
		stride_(Core::Configuration::config(paramStride_, prefix_)),
		previousBatchSize_(1),
		padX_(Core::Configuration::config(paramPadX_, prefix_)),
		padY_(Core::Configuration::config(paramPadY_, prefix_))
{
	useBias_ = false;
}

void SpatialPoolingLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	require_gt(gridSize_, 0);
	require_gt(stride_, 0);
}

void SpatialPoolingLayer::setWidth(u32 port, u32 width) {
	Precursor::setWidth(port, width);
	require_ge(width_, gridSize_);
	require_ge(width_, stride_);
	if (padX_ != -1) {
		destWidth_ = 1 + (width_ + (2 * padX_ - (s32)gridSize_)) / stride_;
	}
	else {
		destWidth_ = (u32)ceil((f64)width_ / (f64)stride_);
		padX_ = (s32)ceil((f64)(((s32)destWidth_ - 1) * (s32)stride_ - ((s32)width_ - (s32)gridSize_)) / 2.0);
	}
}

void SpatialPoolingLayer::setHeight(u32 port, u32 height) {
	Precursor::setHeight(port, height);
	require_ge(height_, gridSize_);
	require_ge(height_, stride_);

	if (padY_ != -1) {
		destHeight_ = 1 + (height_ + (2 * padY_ - (s32)gridSize_)) / stride_;
	}
	else {
		destHeight_ = (u32)ceil((f64)height_ / (f64)stride_);
		padY_ = (s32)ceil((f64)(((s32)destHeight_ - 1) * (s32)stride_ - ((s32)height_ - (s32)gridSize_)) / 2.0);
	}
}

void SpatialPoolingLayer::updateNumberOfUnits(u32 port) {
	Precursor::updateNumberOfUnits(port);
	nOutUnits_ = destHeight_ * destWidth_ * nChannels_;
}

/*
 * MaxPoolingLayer
 */
MaxPoolingLayer::MaxPoolingLayer(const char* name):
		Precursor(name),
		cudnnPooling_(Math::cuDNN::MaxPooling)
{
	useBias_ = false;
}

void MaxPoolingLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);

	if(useCudnn_) {
#ifdef MODULE_CUDNN
		cudnnPooling_.init(gridSize_, stride_, padX_, padY_,
			1, width_, height_, nChannels_, destWidth_, destHeight_, nChannels_);
#endif
	}
}

void MaxPoolingLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;

	if (useCudnn_) {
#ifdef MODULE_CUDNN
		if (previousBatchSize_ != activationsIn(t, port).nColumns()) {
			cudnnPooling_.updateBatchSize(activationsIn(t, port).nColumns());
			previousBatchSize_ = activationsIn(t, port).nColumns();
		}
		cudnnPooling_.poolingForward(activationsOut(t, port), activationsIn(t, port));
#else
		activationsOut(t, port).maxPool(activationsIn(t,port), width_, height_,
				nChannels_, gridSize_, stride_);
#endif
	}
	else {
		activationsOut(t, port).maxPool(activationsIn(t,port), width_, height_,
				nChannels_, gridSize_, stride_);
	}
}

void MaxPoolingLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	if (useCudnn_) {
#ifdef MODULE_CUDNN
	cudnnPooling_.poolingBackward(errorSignalIn(timeframe, port),
			activationsIn(timeframe, port), errorSignalOut(timeframe, port), activationsOut(timeframe, port));
#else
	errorSignalIn(timeframe, port).backPropogateMaxPool(activationsIn(timeframe, port), activationsOut(timeframe, port), errorSignalOut(timeframe, port),
			width_, height_, nChannels_, gridSize_, stride_);
#endif
	}
	else {
		errorSignalIn(timeframe, port).backPropogateMaxPool(activationsIn(timeframe, port), activationsOut(timeframe, port), errorSignalOut(timeframe, port),
					width_, height_, nChannels_, gridSize_, stride_);
	}
}

/*
 * AvgPoolingLayer
 */
AvgPoolingLayer::AvgPoolingLayer(const char* name):
		Precursor(name),
		cudnnPooling_(Math::cuDNN::AvgPooling)
{
	useBias_ = false;
}

void AvgPoolingLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
#ifdef MODULE_CUDNN
	cudnnPooling_.init(gridSize_, stride_, padX_, padY_,
			1, width_, height_, nChannels_, destWidth_, destHeight_, nChannels_);
#endif
}

void AvgPoolingLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
#ifdef MODULE_CUDNN
	if (previousBatchSize_ != activationsIn(t, port).nColumns()) {
		cudnnPooling_.updateBatchSize(activationsIn(t, port).nColumns());
		previousBatchSize_ = activationsIn(t, port).nColumns();
	}
	cudnnPooling_.poolingForward(activationsOut(t, port), activationsIn(t, port));
#else
	activationsOut(t, port).avgPool(activationsIn(t,port), width_, height_,
			nChannels_, gridSize_, stride_);
#endif
}

void AvgPoolingLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
#ifdef MODULE_CUDNN
	cudnnPooling_.poolingBackward(errorSignalIn(timeframe, port),
				activationsIn(timeframe, port), errorSignalOut(timeframe, port), activationsOut(timeframe, port));
#else
	errorSignalIn(timeframe, port).backPropogateAvgPool(errorSignalOut(timeframe, port),
			width_, height_, nChannels_, gridSize_, stride_);
#endif
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

/*
 * AttentionLayer
 */
const Core::ParameterInt AttentionLayer::paramAttentionRange_("attention-range", Types::max<s32>(), "neural-network.layer");

AttentionLayer::AttentionLayer(const char* name) :
		Precursor(name),
		attentionRange_(Core::Configuration::config(paramAttentionRange_, prefix_))
{
	nPorts_ = 2;
	require_gt(attentionRange_, 0);
}

void AttentionLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	require_gt(nIncomingConnections(0), 0);
	require_gt(nIncomingConnections(1), 0);
	tmpMatA_.initComputation();
	tmpMatB_.initComputation();
	tmpVec_.initComputation();
}

void AttentionLayer::blockPorts() {
	blockedOutputPorts_.resize(nOutputPorts(), false);
	blockedOutputPorts_.back() = true;
}

void AttentionLayer::unblockPorts() {
	blockedOutputPorts_.resize(nOutputPorts(), false);
	blockedOutputPorts_.back() = false;
}

bool AttentionLayer::isOutputPortBlocked(u32 port) {
	if ((blockedOutputPorts_.size() > port) && (blockedOutputPorts_.at(port) == true))
		return true;
	else
		return false;
}

void AttentionLayer::setNumberOfChannels(u32 port, u32 channels) {
	// do nothing for attention port 1
	if (port == 0)
		Precursor::setNumberOfChannels(port, channels);
}

// attention port 1 needs to be initialized with -inf
void AttentionLayer::addTimeframe(u32 minibatchSize) {
	Precursor::addTimeframe(minibatchSize);
	latestActivations(1).fill(-Types::inf<Float>());
}

void AttentionLayer::forward() {
	Precursor::forward();
	u32 t = nTimeframes() - 1;
	u32 t_start = (nTimeframes() > attentionRange_ ? nTimeframes() - attentionRange_ : 0);

	// copy attention input for old timeframes
	if ((t > 0) && (t_start == 0))
		activationsOut(t, 1).copyBlockFromMatrix(tmpMatA_, 0, 0, 0, 0, tmpMatA_.nRows(), tmpMatA_.nColumns());
	else if ((t > 0) && (t_start > 0))
		activationsOut(t, 1).copyBlockFromMatrix(tmpMatA_, 1, 0, 0, 0, tmpMatA_.nRows()-1, tmpMatA_.nColumns());
	// copy attention input for current timeframe (scalar values at input port 1)
	activationsOut(t, 1).copyBlockFromMatrix(activationsIn(t, 1), 0, 0, t-t_start, 0, 1, activationsIn(t, 1).nColumns());
	tmpMatA_.resize(activationsOut(t, 1).nRows(), activationsOut(t, 1).nColumns());
	tmpMatA_.copy(activationsOut(t, 1));
	activationsOut(t, 1).softmax();
	for (u32 i = t_start; i <= t; i++) {
		activationsOut(t, 1).getRow(i-t_start, tmpVec_);
		tmpVec_.safeResize(activationsIn(i, 0).nColumns());
		activationsOut(t, 0).addWeighted(activationsIn(i, 0), tmpVec_);
	}
}

void AttentionLayer::backpropagate(u32 timeframe) {
	u32 t_start = (timeframe + 1 > attentionRange_ ? timeframe + 1 - attentionRange_ : 0);
	Precursor::backpropagate(timeframe);
	// error signal wrt. port 1 (attentions, 1-dimensional)
	if (t_start == 0)
		tmpMatA_.resize(timeframe+1, latestActivations(0).nColumns());
	else
		tmpMatA_.resize(attentionRange_, latestActivations(0).nColumns());
	tmpMatA_.setToZero();
	for (u32 t = t_start; t <= timeframe; t++) {
		errorSignalOut(timeframe, 0).setVisibleColumns(activationsIn(t, 0).nColumns());
		// compute columnwise inner product (much faster this way than with the Matrix function)
		tmpVec_.resize(activationsIn(t, 0).nColumns());
		tmpMatB_.copyStructure(errorSignalOut(timeframe, 0));
		tmpMatB_.copy(activationsIn(t, 0));
		tmpMatB_.elementwiseMultiplication(errorSignalOut(timeframe, 0));
		tmpVec_.setToZero();
		tmpVec_.addSummedRows(tmpMatB_);
		// copy the result to tmpMatA_ at the position of the respective timeframe
		tmpMatA_.safeResize(tmpMatA_.nRows(), tmpVec_.size());
		tmpMatA_.setRow(t-t_start, tmpVec_);
	}
	tmpMatA_.multiplicationWithSoftmaxDerivative(activationsOut(timeframe, 1));
	for (u32 t = t_start; t <= timeframe; t++) {
		errorSignalIn(t, 1).addBlockFromMatrix(tmpMatA_, t-t_start, 0, 0, 0, 1, errorSignalIn(t, 1).nColumns());
	}
	u32 T = (nTimeframes() - timeframe > attentionRange_ ? timeframe + attentionRange_ : nTimeframes());
	// error signal wrt. port 0 (recurrent layer outputs)
	for (u32 t = timeframe; t < T; t++) {
		tmpMatA_.copyStructure(errorSignalIn(timeframe, 0));
		errorSignalOut(t, 0).safeResize(tmpMatA_.nRows(), tmpMatA_.nColumns());
		tmpMatA_.copy(errorSignalOut(t, 0));
		activationsOut(t, 1).getRow(timeframe-t_start, tmpVec_);
		tmpVec_.resize(tmpMatA_.nColumns());
		tmpMatA_.multiplyColumnsByScalars(tmpVec_);
		errorSignalIn(timeframe, 0).add(tmpMatA_);
	}
}

/*
 * Batch Normalization
 */
const Core::ParameterBool BatchNormalizationLayer::paramIsSpatial_("is-spatial", true, "neural-network.layer");

const Core::ParameterBool BatchNormalizationLayer::paramIsInference_("is-inference", false, "neural-network.layer");

BatchNormalizationLayer::BatchNormalizationLayer(const char* name):
		Precursor(name),
		isSpatial_(Core::Configuration::config(paramIsSpatial_, prefix_)),
		isInference_(Core::Configuration::config(paramIsInference_, prefix_)),
		cudnnBatchNormalization_(isSpatial_ ? Math::cuDNN::Spatial : Math::cuDNN::PerActivation),
		nIterations_(1),
		prevBatchSize_(2)
{
	useBias_ = false;
}

void BatchNormalizationLayer::initializeParam(Vector &vector, const std::string& basePath,
		const std::string& suffix, const std::string& paramName, ParamInitialization initMethod) {
	struct stat buffer;
	if (suffix.compare("") != 0 &&
			stat(getParamFileName(basePath, paramName, suffix, 0).c_str(), &buffer) == 0) {
		std::string filename = getParamFileName(basePath, paramName, suffix, 0);
		Core::Log::os("Layer ") << name_ << ":" << ": read "<<paramName<<" from " << filename;
		vector.read(filename);
	}
	else if (stat(getParamFileName(basePath, paramName, "", 0).c_str(), &buffer) == 0) {
		std::string filename = getParamFileName(basePath, paramName, "", 0);
		Core::Log::os("Layer ") << name_ << ":" << ": read "<<paramName<<" from " << filename;
		vector.read(filename);
	}
	else {
		_initializeParam(vector, initMethod);
	}
}

void BatchNormalizationLayer::initializeParams(const std::string& basePath, const std::string& suffix) {
	require(!isComputing_);

	bias_.resize(1);
	bias_.at(0).resize(2 * nChannels_);

	gamma_.resize(nChannels_);
	beta_.resize(nChannels_);
	gammaDer_.resize(nChannels_ );
	betaDer_.resize(nChannels_ );
	runningMean_.resize(nChannels_);
	runningVariance_.resize(nChannels_);
	saveMean_.resize(nChannels_);
	saveVariance_.resize(nChannels_);

	gammaDer_.setToZero();
	betaDer_.setToZero();
	saveMean_.setToZero();
	saveVariance_.setToZero();

	initializeParam(gamma_, basePath, suffix, "gamma", random);
	initializeParam(beta_, basePath, suffix, "beta", random);
	initializeParam(runningMean_, basePath, suffix, "running-mean", zero);
	initializeParam(runningVariance_, basePath, suffix, "running-variance", zero);
}

void BatchNormalizationLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	Precursor::initialize(basePath, suffix, maxMemory);
	initializeParams(basePath, suffix);
	cudnnBatchNormalization_.init(2, nChannels_, height_, width_, nChannels_, destHeight_, destWidth_);
}

void BatchNormalizationLayer::initComputation(bool sync) {
	if (isComputing_)
		return;

	Precursor::initComputation(sync);

	bias_.at(0).initComputation(sync);
	gamma_.initComputation(sync);
	beta_.initComputation(sync);
	runningMean_.initComputation(sync);
	runningVariance_.initComputation(sync);
	saveMean_.initComputation(sync);
	saveVariance_.initComputation(sync);
	gammaDer_.initComputation(sync);
	betaDer_.initComputation(sync);

	bias_.at(0).copyBlockFromVector(gamma_, 0, 0, nChannels_);
	bias_.at(0).copyBlockFromVector(beta_, 0, nChannels_, nChannels_);
}

void BatchNormalizationLayer::finishComputation(bool sync) {
	if (!isComputing_)
		return;

	Precursor::finishComputation(sync);

	gamma_.copyBlockFromVector(bias_.at(0), 0, 0, nChannels_);
	beta_.copyBlockFromVector(bias_.at(0), nChannels_, 0, nChannels_);

	bias_.at(0).finishComputation(sync);
	gamma_.finishComputation(sync);
	beta_.finishComputation(sync);
	runningMean_.finishComputation(sync);
	runningVariance_.finishComputation(sync);
	saveMean_.finishComputation(sync);
	saveVariance_.finishComputation(sync);
	gammaDer_.finishComputation(sync);
	betaDer_.finishComputation(sync);
}

void BatchNormalizationLayer::save(Vector &vector, const std::string& basePath,
		const std::string& suffix, const std::string& paramName) {
	std::string fn = getParamFileName(basePath, paramName, suffix, 0);
	bool isBiasComputing = vector.isComputing();
	vector.finishComputation();
	vector.write(fn);
	if (isBiasComputing)
		vector.initComputation(false);
}

void BatchNormalizationLayer::saveParams(const std::string& basePath, const std::string& suffix) {
	Precursor::saveParams(basePath, suffix);
	gamma_.copyBlockFromVector(bias_.at(0), 0, 0, nChannels_);
	beta_.copyBlockFromVector(bias_.at(0), nChannels_, 0, nChannels_);

	save(beta_, basePath, suffix, "beta");
	save(gamma_, basePath, suffix, "gamma");
	save(runningMean_, basePath, suffix, "running-mean");
	save(runningVariance_, basePath, suffix, "running-variance");
}

void BatchNormalizationLayer::forward(u32 port) {
	Precursor::forward(port);
	u32 t = nTimeframes() - 1;
#ifdef MODULE_CUDNN
	if (prevBatchSize_ != activationsIn(t, port).nColumns()) {
		prevBatchSize_ = activationsIn(t, port).nColumns();
		cudnnBatchNormalization_.updateBatchSize(prevBatchSize_);
	}
	if(!isInference_) {
		nIterations_++;
		cudnnBatchNormalization_.batchNormalizationForward(activationsOut(t, port), activationsIn(t, port), gamma_, beta_,
				(1.0/nIterations_), runningMean_, runningVariance_, saveMean_, saveVariance_);
	}
	else {
		cudnnBatchNormalization_.batchNormalizationForwardInference(activationsOut(t, port), activationsIn(t, port),
				gamma_, beta_, runningMean_, runningVariance_);
	}

#else
	std::cerr<<"cudnn is required for BatchNormalization"<<std::endl;
	exit(1);
#endif
}

void BatchNormalizationLayer::backpropagate(u32 timeframe, u32 port) {
	Precursor::backpropagate(timeframe, port);
	if (nOutgoingConnections(port) > 0) {
#ifdef MODULE_CUDNN
		gamma_.copyBlockFromVector(bias_.at(0), 0, 0, nChannels_);
		beta_.copyBlockFromVector(bias_.at(0), nChannels_, 0, nChannels_);

		cudnnBatchNormalization_.batchNormalizationBackward(errorSignalIn(timeframe, port), errorSignalOut(timeframe, port),
				activationsIn(timeframe, port), gamma_, gammaDer_, betaDer_, saveMean_, saveVariance_);
#else
	std::cerr<<"cudnn is required for BatchNormalization"<<std::endl;
	exit(1);
#endif
	}
}

void BatchNormalizationLayer::getBiasGradient(Vector &biasGradient) {
	biasGradient.finishComputation();
	biasGradient.resize(2 * nChannels_);
	biasGradient.initComputation();

	biasGradient.copyBlockFromVector(gammaDer_, 0, 0, nChannels_);
	biasGradient.copyBlockFromVector(betaDer_, 0, nChannels_, nChannels_);
}

void BatchNormalizationLayer::updateParams(f32 learningRate) {
#ifdef MODULE_CUDNN
	gamma_.add(gammaDer_, learningRate);
	beta_.add(betaDer_, learningRate);
#else
	std::cerr<<"cudnn is required for BatchNormalization"<<std::endl;
	exit(1);
#endif
}

/*
 * Pre-Processing layer
 * Creates a stack of frames from given n frames
 */

const Core::ParameterInt PreProcessingLayer::paramTotalFrames_("total-frames", 1, "neural-network.layer");

PreProcessingLayer::PreProcessingLayer(const char* name):
		Precursor(name),
		totalFrames_(Core::Configuration::config(paramTotalFrames_, prefix_)) {
	useBias_ = false;
	destChannels_ = totalFrames_;
	isRecurrent_ = true;
}

u32 PreProcessingLayer::nOutputUnits(u32 port) const {
	return destChannels_ * nInputUnits(port);
}
void PreProcessingLayer::updateNumberOfUnits(u32 port) {
	Precursor::updateNumberOfUnits(port);
	nOutUnits_ = nOutputUnits(port);
}

void PreProcessingLayer::finalizeForwarding() {
	for (u32 i = 0; i < totalFrames_; i++) {
		activationsOut(0, 0).copyBlockFromMatrix(activationsIn(i, 0), 0, 0,
				i * activationsIn(i, 0).nRows(), 0, activationsIn(i, 0).nRows(), activationsIn(i, 0).nColumns());
	}
}

void PreProcessingLayer::setMaximalMemory(u32 maxMemory) {
	Precursor::setMaximalMemory(totalFrames_);
}

void PreProcessingLayer::backpropagate(u32 timeframe, u32 port) {
	std::cerr<<"Backpropagation is not implemented for PreProcessing Layer"<<std::endl;
}
