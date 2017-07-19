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
 * FeatureTransformationLayer.cc
 *
 *  Created on: Jun 6, 2014
 *      Author: richard
 */

#include "FeatureTransformationLayer.hh"

using namespace Nn;

/*
 * FeatureTransformationLayer
 */
FeatureTransformationLayer::FeatureTransformationLayer(const char* name) :
		Precursor(name)
{
	useBias_ = false;
	isBiasTrainable_ = false;
}

void FeatureTransformationLayer::addIncomingConnection(Connection* c, u32 port) {
	if (nIncomingConnections(port) > 0) {
		std::cerr << "FeatureTransformationLayer: number of incoming connections can not be larger than 1. Abort." << std::endl;
		exit(1);
	}
	if (c->hasWeights()) {
		std::cerr << "FeatureTransformationLayer: incoming connection must be of type plain-connection. Abort." << std::endl;
		exit(1);
	}
	Precursor::addIncomingConnection(c, port);
}

void FeatureTransformationLayer::backpropagate(u32 timeframe, u32 port) {
	std::cerr << "FeatureTransformationLayer::backpropagate is not implemented or not possible for feature transformation layer "
			<< name_ << ". Abort." << std::endl;
	exit(1);
}

/*
 * PolynomialPreprocessingLayer
 */
const Core::ParameterInt PolynomialPreprocessingLayer::paramPolynomialOrder_("order", 1, "neural-network.layer");

// if true use only "diagonal" parts of the polynomial expansion, e.g. x_1^2, but not x_1*x_2
const Core::ParameterBool PolynomialPreprocessingLayer::paramOnlyDiagonalElements_("use-only-diagonal-elements",
		false, "neural-network.layer");

const Core::ParameterInt PolynomialPreprocessingLayer::paramBaseDimension_("base-dimension", 0, "neural-netowork.layer");

PolynomialPreprocessingLayer::PolynomialPreprocessingLayer(const char* name) :
		Precursor(name),
		order_(Core::Configuration::config(paramPolynomialOrder_, prefix_)),
		onlyDiagonalElements_(Core::Configuration::config(paramOnlyDiagonalElements_, prefix_)),
		baseDimension_(Core::Configuration::config(paramBaseDimension_, prefix_))
{
	require_ge(order_, 1);
	if (baseDimension_ == 0) {
		std::cerr << "PolynomialPreprocessingLayer: "
				<< "base-dimension needs to be equal to the output dimension of the preceding layer. Abort." << std::endl;
		exit(1);
	}
}

void PolynomialPreprocessingLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	nUnits_ = baseDimension_;
	if (order_ > 1)
		nUnits_ += onlyDiagonalElements_ ? baseDimension_ : baseDimension_ * (baseDimension_ + 1) / 2;
	if (order_ > 2)
		nUnits_ += onlyDiagonalElements_ ? baseDimension_ : baseDimension_ * (baseDimension_ + 1) * (baseDimension_ + 2) / 6;
	if (order_ > 3) {
		std::cerr << "PolynomialPreprocessingLayer: polynomial order > 3 not implemented. Abort." << std::endl;
		exit(1);
	}
	Precursor::initialize(basePath, suffix, maxMemory);
}

void PolynomialPreprocessingLayer::forward(u32 port) {
	require(nIncomingConnections(port) == 1);
	require_eq(baseDimension_, incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()));
	require(isComputing_);
	require_ge(nTimeframes(), 1);

	u32 t = nTimeframes() - 1;
	u32 sourcePort = incomingConnection(0, port).sourcePort();
	if (order_ == 1)
		activationsIn(t, port).copy(incomingConnection(0, port).from().activationsOut(t, sourcePort));
	if ((order_ == 2) && onlyDiagonalElements_)
		activationsIn(t, port).setToDiagonalSecondOrderFeatures(incomingConnection(0, port).from().activationsOut(t, sourcePort));
	if ((order_ == 2) && (!onlyDiagonalElements_))
		activationsIn(t, port).setToSecondOrderFeatures(incomingConnection(0, port).from().activationsOut(t, sourcePort));
	if ((order_ == 3) && onlyDiagonalElements_)
		activationsIn(t, port).setToDiagonalThirdOrderFeatures(incomingConnection(0, port).from().activationsOut(t, sourcePort));
	if ((order_ == 3) && (!onlyDiagonalElements_))
		activationsIn(t, port).setToThirdOrderFeatures(incomingConnection(0, port).from().activationsOut(t, sourcePort));
}

/*
 * FisherLayer
 */
const Core::ParameterString FisherLayer::paramMeanFile_("mean-file", "", "neural-network.layer");

const Core::ParameterString FisherLayer::paramVarianceFile_("variance-file", "", "neural-network.layer");

const Core::ParameterString FisherLayer::paramWeightsFile_("weights-file", "", "neural-network.layer");

FisherLayer::FisherLayer(const char* name) :
		Precursor(name),
		meanFile_(Core::Configuration::config(paramMeanFile_, prefix_)),
		varianceFile_(Core::Configuration::config(paramVarianceFile_, prefix_)),
		weightsFile_(Core::Configuration::config(paramWeightsFile_, prefix_))
{}

void FisherLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	require(!meanFile_.empty());
	require(!varianceFile_.empty());
	require(!weightsFile_.empty());
	means_.read(meanFile_);
	variances_.read(varianceFile_);
	weights_.read(weightsFile_);
	require_eq(means_.nRows(), variances_.nRows());
	require_eq(means_.nColumns(), variances_.nColumns());
	require_eq(means_.nRows(), weights_.nRows());
	means_.initComputation();
	variances_.initComputation();
	weights_.initComputation();
	nUnits_ = means_.nColumns() * means_.nRows() * 2;
	Precursor::initialize(basePath, suffix, maxMemory);
}

void FisherLayer::forward(u32 port) {
	require(nIncomingConnections(port) == 1);
	require_eq(means_.nColumns(), incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()));
	require(isComputing_);
	require_ge(nTimeframes(), 1);

	u32 t = nTimeframes() - 1;
	u32 sourcePort = incomingConnection(0, port).sourcePort();
	activationsIn(t, port).fisherEncoding(incomingConnection(0, port).from().activationsOut(t, sourcePort), means_, variances_, weights_);
}

/*
 * FeatureCloningLayer
 */
const Core::ParameterEnum FeatureCloningLayer::paramCloningMode_("cloning-mode",
		"on-the-whole, elementwise", "on-the-whole", "neural-network.layer");

const Core::ParameterInt FeatureCloningLayer::paramNumberOfClones_("number-of-clones",
		1, "neural-network.layer");

FeatureCloningLayer::FeatureCloningLayer(const char* name) :
		Precursor(name),
		cloningMode_((CloningMode) Core::Configuration::config(paramCloningMode_, prefix_)),
		nClones_(Core::Configuration::config(paramNumberOfClones_, prefix_))
{}

void FeatureCloningLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	for (u32 port = 0; port < nInputPorts(); port++) {
		require_eq(nIncomingConnections(port), 1);
		require_eq(incomingConnection(0, port).type(), Connection::plainConnection);
	}
	nUnits_ = incomingConnection(0, 0).from().nOutputUnits(incomingConnection(0, 0).sourcePort()) * nClones_;
	Precursor::initialize(basePath, suffix, maxMemory);
}

void FeatureCloningLayer::forward(u32 port) {
	u32 t = nTimeframes() - 1;
	u32 sourcePort = incomingConnection(0, port).sourcePort();
	const Matrix& activations = incomingConnection(0, port).from().activationsOut(t, sourcePort);
	activationsIn(t, port).resize(activations.nRows() * nClones_, activations.nColumns());
	if (cloningMode_ == onTheWhole) {
		activationsIn(t, port).clone(activations, nClones_);
	}
	else { // if cloningMode_ == elementwise
		activationsIn(t, port).cloneElementwise(activations, nClones_);
	}
}

void FeatureCloningLayer::backpropagate(u32 timeframe, u32 port) {
	Layer::backpropagate(timeframe, port);
	u32 nRows = errorSignalOut(timeframe, port).nRows() / nClones_;
	u32 nColumns = errorSignalOut(timeframe, port).nColumns();
	tmpMatrix_.resize(nRows, nColumns);
	tmpMatrix_.setToZero();
	if (cloningMode_ == onTheWhole) {
		tmpMatrix_.addElementsByModuloIndex(errorSignalOut(timeframe, port));
	}
	else { // if cloningMode_ == elementwise
		errorSignalOut(timeframe, port).safeResize(nClones_, nRows * nColumns);
		Vector tmp;
		tmp.initComputation();
		tmp.swap(tmpMatrix_);
		tmp.addSummedRows(errorSignalOut(timeframe, port));
		tmpMatrix_.swap(tmp);
		tmpMatrix_.safeResize(nRows, nColumns);
	}
	tmpMatrix_.swap(errorSignalOut(timeframe, port));
}

void FeatureCloningLayer::initComputation(bool sync) {
	tmpMatrix_.initComputation(sync);
	Precursor::initComputation(sync);
}

void FeatureCloningLayer::finishComputation(bool sync) {
	tmpMatrix_.finishComputation(sync);
	Precursor::finishComputation(sync);
}

/*
 * ApproximateFeatureMapLayer
 */
const Core::ParameterInt ApproximateFeatureMapLayer::paramSamplingPointsPerFeature_("sampling-points-per-feature", 0, "neural-network.layer");

const Core::ParameterFloat ApproximateFeatureMapLayer::paramSamplingDistance_("sampling-distance", 0.5, "neural-network.layer");

const Core::ParameterEnum ApproximateFeatureMapLayer::paramKernelType_("kernel-type", "chi-square, histogram-intersection", "chi-square",
		"neural-network.layer");

ApproximateFeatureMapLayer::ApproximateFeatureMapLayer(const char* name) :
		Precursor(name),
		nSamples_(Core::Configuration::config(paramSamplingPointsPerFeature_, prefix_)),
		samplingDistance_(Core::Configuration::config(paramSamplingDistance_, prefix_)),
		kernelType_((KernelType)Core::Configuration::config(paramKernelType_, prefix_))
{}

void ApproximateFeatureMapLayer::initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory) {
	for (u32 port = 0; port < nInputPorts(); port++) {
		require_eq(nIncomingConnections(port), 1);
		require_eq(incomingConnection(0, port).type(), Connection::plainConnection);
	}
	nUnits_ = incomingConnection(0, 0).from().nOutputUnits(incomingConnection(0, 0).sourcePort()) * (2 * nSamples_ + 1);
	Precursor::initialize(basePath, suffix, maxMemory);
}

void ApproximateFeatureMapLayer::forward(u32 port) {
	require(nIncomingConnections(port) == 1);
	require(isComputing_);
	require_ge(nTimeframes(), 1);

	u32 t = nTimeframes() - 1;
	u32 sourcePort = incomingConnection(0, port).sourcePort();
	switch (kernelType_) {
	case chiSquare:
		activationsIn(t, port).chiSquareFeatureMap(incomingConnection(0, port).from().activationsOut(t, sourcePort), nSamples_, samplingDistance_);
		break;
	case histogramIntersection:
		activationsIn(t, port).histogramIntersectionFeatureMap(incomingConnection(0, port).from().activationsOut(t, sourcePort), nSamples_, samplingDistance_);
		break;
	default:
		; // this can not happen
	}
}

void ApproximateFeatureMapLayer::backpropagate(u32 timeframe, u32 port) {
	Layer::backpropagate(timeframe, port);
	// compute element-wise derivative
	switch (kernelType_) {
	case chiSquare:
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithApproximateFeatureMapDerivative(activationsOut(timeframe, port),
					nSamples_, samplingDistance_, 1.0);
		break;
	case histogramIntersection:
		errorSignalOut(timeframe, port).elementwiseMultiplicationWithApproximateFeatureMapDerivative(activationsOut(timeframe, port),
					nSamples_, samplingDistance_, 2.0 / M_PI);
		break;
	default:
		; // this can not happen
	}
	// add the derivatives that are taken with respect to the same feature
	tmpMatrix_.resize(incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()), errorSignalOut(timeframe, port).nColumns());
	tmpMatrix_.setToZero();
	tmpMatrix_.addSummedNeighborsInARow(errorSignalOut(timeframe, port), 2 * nSamples_ + 1);
	// store the result in the error signals container
	errorSignalOut(timeframe, port).swap(tmpMatrix_);
}

void ApproximateFeatureMapLayer::initComputation(bool sync) {
	tmpMatrix_.initComputation(sync);
	Precursor::initComputation(sync);
}

void ApproximateFeatureMapLayer::finishComputation(bool sync) {
	tmpMatrix_.finishComputation(sync);
	Precursor::finishComputation(sync);
}

/*
 * ModulatedSumLayer
 */
ModulatedSumLayer::ModulatedSumLayer(const char* name) :
		Precursor(name)
{}

void ModulatedSumLayer::forward(u32 port) {
	require(nIncomingConnections(port) == 1);
	require_eq(incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()) % nInputUnits(port), 0);
	require(isComputing_);
	require_ge(nTimeframes(), 1);

	u32 t = nTimeframes() - 1;
	u32 sourcePort = incomingConnection(0, port).sourcePort();
	activationsIn(t, port).setToZero();
	activationsIn(t, port).addElementsByModuloIndex(incomingConnection(0, port).from().activationsOut(t, sourcePort));
}

void ModulatedSumLayer::backpropagate(u32 timeframe, u32 port) {
	require_eq(incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()) % nInputUnits(port), 0);
	Layer::backpropagate(timeframe, port);
	// add the derivatives that are taken with respect to the same feature
	tmpMatrix_.resize(incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()), errorSignalOut(timeframe, port).nColumns());
	u32 n = incomingConnection(0, port).from().nOutputUnits(incomingConnection(0, port).sourcePort()) / nInputUnits(port);
	for (u32 i = 0; i < n; i++) {
		tmpMatrix_.copyBlockFromMatrix(errorSignalOut(timeframe, port),
				0, 0,
				i * errorSignalOut(timeframe, port).nRows(), 0,
				errorSignalOut(timeframe, port).nRows(), errorSignalOut(timeframe, port).nColumns()
				);
	}
	// store the result in the error signals container
	errorSignalOut(timeframe, port).swap(tmpMatrix_);
}

void ModulatedSumLayer::initComputation(bool sync) {
	tmpMatrix_.initComputation(sync);
	Precursor::initComputation(sync);
}

void ModulatedSumLayer::finishComputation(bool sync) {
	tmpMatrix_.finishComputation(sync);
	Precursor::finishComputation(sync);
}
