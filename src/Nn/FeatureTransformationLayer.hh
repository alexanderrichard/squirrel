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
 * FeatureTransformationLayer.hh
 *
 *  Created on: Jun 6, 2014
 *      Author: richard
 */

#ifndef NN_FEATURETRANSFORMATIONLAYER_HH_
#define NN_FEATURETRANSFORMATIONLAYER_HH_

#include <Core/CommonHeaders.hh>
#include "Layer.hh"

namespace Nn {

/*
 * base class for feature transformation layers
 */
class FeatureTransformationLayer : public Layer
{
private:
	typedef Layer Precursor;
public:
	FeatureTransformationLayer(const char* name);
	virtual ~FeatureTransformationLayer() {}
	virtual void addIncomingConnection(Connection* c, u32 port = 0);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * polynomial feature expansion
 */
class PolynomialPreprocessingLayer : public FeatureTransformationLayer {
private:
	typedef FeatureTransformationLayer Precursor;
	static const Core::ParameterInt paramPolynomialOrder_;
	static const Core::ParameterBool paramOnlyDiagonalElements_;
	static const Core::ParameterInt paramBaseDimension_;
private:
	u32 order_;
	bool onlyDiagonalElements_;
	u32 baseDimension_;
public:
	PolynomialPreprocessingLayer(const char* name);
	virtual ~PolynomialPreprocessingLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward(u32 port);
};

/*
 * Fisher layer
 * computes fisher encoding of an input vector (without l2- and power-normalization)
 */
class FisherLayer : public FeatureTransformationLayer {
private:
	typedef FeatureTransformationLayer Precursor;
	static const Core::ParameterString paramMeanFile_;
	static const Core::ParameterString paramVarianceFile_;
	static const Core::ParameterString paramWeightsFile_;
private:
	std::string meanFile_;
	std::string varianceFile_;
	std::string weightsFile_;
	Matrix means_;
	Matrix variances_;
	Vector weights_;
public:
	FisherLayer(const char* name);
	virtual ~FisherLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward(u32 port);
};

/*
 * feature cloning layer
 */
class FeatureCloningLayer : public FeatureTransformationLayer {
private:
	typedef FeatureTransformationLayer Precursor;
	static const Core::ParameterEnum paramCloningMode_;
	static const Core::ParameterInt paramNumberOfClones_;
	enum CloningMode { onTheWhole, elementwise };
private:
	CloningMode cloningMode_;
	u32 nClones_;
	Matrix tmpMatrix_;
public:
	FeatureCloningLayer(const char* name);
	virtual ~FeatureCloningLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * approximate feature map layer
 */
class ApproximateFeatureMapLayer : public FeatureTransformationLayer {
private:
	typedef FeatureTransformationLayer Precursor;
	static const Core::ParameterInt paramSamplingPointsPerFeature_;
	static const Core::ParameterFloat paramSamplingDistance_;
	static const Core::ParameterEnum paramKernelType_;
	enum KernelType { chiSquare, histogramIntersection };
private:
	u32 nSamples_;
	Float samplingDistance_;
	KernelType kernelType_;
	Matrix tmpMatrix_;
public:
	ApproximateFeatureMapLayer(const char* name);
	virtual ~ApproximateFeatureMapLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * modulated sum layer (sum all elements of preceeding layer that have the same modulo value)
 */
class ModulatedSumLayer : public FeatureTransformationLayer {
	typedef FeatureTransformationLayer Precursor;
private:
	Matrix tmpMatrix_;
public:
	ModulatedSumLayer(const char* name);
	virtual ~ModulatedSumLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_FEATURETRANSFORMATIONLAYER_HH_ */
