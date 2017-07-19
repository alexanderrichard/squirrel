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
 * MultiPortLayer.hh
 *
 *  Created on: May 17, 2016
 *      Author: ahsan
 */

#ifndef NN_MULTIPORTLAYER_HH_
#define NN_MULTIPORTLAYER_HH_
#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Layer.hh"


namespace Nn
{

/*
 * MultiPortLayer
 * base class for all layers having different input and output size or need distinct input and output containers
 */
// TODO rename, MultiPortLayer is not related to the functionality of this layer
class MultiPortLayer: public Layer
{
private:
	typedef Layer Precursor;
	std::vector<MatrixContainer> outputActivations_;
	std::vector<MatrixContainer> outputErrorSignals_;

protected:
	u32 destWidth_;
	u32 destHeight_;
	u32 nOutUnits_;

public:
	MultiPortLayer(const char* name);
	virtual ~MultiPortLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);

	virtual Matrix& activationsOut(u32 timeframe, u32 port);
	virtual Matrix& errorSignalOut(u32 timeframe, u32 port);

	virtual void addTimeframe(u32 minibatchSize);
	virtual void addEmptyTimeframe();
	virtual void resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns);
	virtual void setMaximalMemory(u32 maxMemory);
	virtual void reset();

	virtual void setActivationVisibility(u32 timeframe, u32 nVisibleColumns);
	virtual void setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);

	virtual void setWidth(u32 port, u32 width);
	virtual void setHeight(u32 port, u32 height);
	virtual u32 width(u32 port) const { return destWidth_; }
	virtual u32 height(u32 port) const { return destHeight_; }

	virtual void updateNumberOfUnits(u32 port);
	virtual u32 nOutputUnits(u32 port) const { return nOutUnits_; }
};

/*
 * PortFusionLayer
 * base class to combine all input ports to a single output port
 */
class PortFusionLayer : public MultiPortLayer
{
private:
	typedef MultiPortLayer Precursor;
public:
	PortFusionLayer(const char* name) : Precursor(name) {}
	virtual ~PortFusionLayer() {}
	virtual u32 nOutputPorts() const { return 1; }
};

/*
 * TriangleActivationLayer
 * realizes the activation function f(x) = |x|, if -1 <= x <= 1, else 0
 */
class TriangleActivationLayer : public MultiPortLayer
{
private:
	typedef MultiPortLayer Precursor;
public:
	TriangleActivationLayer(const char* name);
	virtual ~TriangleActivationLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};


/*
 * TemporalAveragingLayer
 */
class TemporalAveragingLayer : public MultiPortLayer
{
private:
	typedef MultiPortLayer Precursor;
public:
	TemporalAveragingLayer(const char* name);
	virtual ~TemporalAveragingLayer() {}

	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * SpatialPoolingLayer
 * Kind of abstract class, shouldn't be instantiated
 */
class SpatialPoolingLayer: public MultiPortLayer
{
private:
	static const Core::ParameterInt paramGridSize_;
	static const Core::ParameterInt paramStride_;
	static const Core::ParameterInt paramPadX_;
	static const Core::ParameterInt paramPadY_;
	typedef MultiPortLayer Precursor;
protected:
	u32 gridSize_;
	u32 stride_;

	u32 previousBatchSize_;

	s32 padX_;
	s32 padY_;
public:
	SpatialPoolingLayer(const char* name);
	virtual ~SpatialPoolingLayer() {}

	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);

	virtual void setWidth(u32 port, u32 width);
	virtual void setHeight(u32 port, u32 height);
	virtual void updateNumberOfUnits(u32 port);
};

/*
 * MaxPoolingLayer
 * performs max-pooling
 */
class MaxPoolingLayer: public SpatialPoolingLayer
{
private:
	typedef SpatialPoolingLayer Precursor;
#ifdef MODULE_CUDNN
	CudnnPooling cudnnPooling_;
#endif
public:
	MaxPoolingLayer(const char* name);
	virtual ~MaxPoolingLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * AvgPoolingLayer
 * Performs spatial average pooling
 */
class AvgPoolingLayer: public SpatialPoolingLayer
{
private:
	typedef SpatialPoolingLayer Precursor;
#ifdef MODULE_CUDNN
	CudnnPooling cudnnPooling_;
#endif
public:
	AvgPoolingLayer(const char* name);
	virtual ~AvgPoolingLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);

	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * MultiplicationLayer
 *
 * multiplies all input ports
 */
class MultiplicationLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
public:
	MultiplicationLayer(const char* name);
	virtual ~MultiplicationLayer() {}

	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * ConcatenationLayer
 *
 * concatenates the inputs at all input ports to a single output
 */
class ConcatenationLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
public:
	ConcatenationLayer(const char* name);
	virtual ~ConcatenationLayer() {}

	virtual void updateNumberOfUnits(u32 port);
	virtual u32 nOutputUnits(u32 port) const;
	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * MaxoutLayer
 */
class MaxoutLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
public:
	MaxoutLayer(const char* name);
	virtual ~MaxoutLayer() {}

	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * GatedRecurrentUnitLayer
 */
class GatedRecurrentUnitLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
	static const Core::ParameterString paramOldGruFilename_;
	static const Core::ParameterString paramNewGruFilename_;
	std::vector<bool> blockedInputPorts_;
public:
	GatedRecurrentUnitLayer(const char* name);
	virtual ~GatedRecurrentUnitLayer() {}

	virtual void blockPorts();
	virtual void unblockPorts();
	virtual bool isInputPortBlocked(u32 port);

	virtual void addInternalConnections(NeuralNetwork& network);
	virtual u32 nInputPorts() const { return 4; }
	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * AttentionLayer
 */
class AttentionLayer : public PortFusionLayer
{
private:
	static const Core::ParameterInt paramAttentionRange_;
	typedef PortFusionLayer Precursor;

	u32 attentionRange_;
	std::vector<bool> blockedOutputPorts_;
	Matrix tmpMatA_;
	Matrix tmpMatB_;
	Vector tmpVec_;
public:
	AttentionLayer(const char* name);
	virtual ~AttentionLayer() {}

	virtual u32 nInputPorts() const { return 2; }
	virtual u32 nOutputPorts() const { return 2; } // output port 0: weighted input activations, output port 1: attentions (variable length)

	// second output port (containing the attentions) needs to be blocked during network creation (else it will be connected to the next layer)
	virtual void blockPorts();
	virtual void unblockPorts();
	virtual bool isOutputPortBlocked(u32 port);
	virtual void setNumberOfChannels(u32 port, u32 channels);

	virtual void addTimeframe(u32 minibatchSize);

	virtual bool isRecurrent() const { return true; }
	virtual u32 nInputUnits(u32 port) const { return (port == 0 ? nUnits_ : 1); }
	virtual u32 nOutputUnits(u32 port) const { return (port == 0 ? nUnits_ : std::min(attentionRange_, nTimeframes())); }
	virtual u32 nChannels(u32 port) const { return (port == 0 ? nChannels_ : 1); }

	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * BatchNormalization layer
 */
class BatchNormalizationLayer : public MultiPortLayer
{
private:
	typedef MultiPortLayer Precursor;
	static const Core::ParameterBool paramIsSpatial_;
	static const Core::ParameterBool paramIsInference_;
	bool isSpatial_;
	bool isInference_;

	CudnnBatchNormalization cudnnBatchNormalization_;

	Vector gamma_;
	Vector beta_;
	Vector runningMean_;
	Vector runningVariance_;
	Vector saveMean_;
	Vector saveVariance_;
	Vector gammaDer_;
	Vector betaDer_;
	u32 nIterations_;
	u32 prevBatchSize_;


private:
	void save(Vector& vector, const std::string& basePath, const std::string& suffix, const std::string& paramName);
	void initializeParam(Vector& vector, const std::string& basePath,
			const std::string& suffix, const std::string& paramName, ParamInitialization initMethod);
protected:
	void initializeParams(const std::string& basePath, const std::string& suffix);

public:
	BatchNormalizationLayer(const char* name);
	virtual ~BatchNormalizationLayer() {}
	virtual void initialize(const std::string& basePath, const std::string& suffix, u32 maxMemory = 1);
	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);

	virtual void saveParams(const std::string& basePath, const std::string& suffix);

	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);

	virtual void updateParams(f32 learningRate);
	virtual bool isBiasTrainable() const { return true; }
	void getBiasGradient(Vector &biasGradient);
};
/*
 * pre-processing layer
 */
class PreProcessingLayer: public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
	static const Core::ParameterInt paramTotalFrames_;

	u32 totalFrames_;

protected:
	u32 destChannels_;

public:
	PreProcessingLayer(const char* name);
	virtual ~PreProcessingLayer() {}

	virtual u32 nChannels(u32 port) const { return destChannels_; }
	virtual void setMaximalMemory(u32 maxMemory);

	virtual void finalizeForwarding();

	virtual u32 nOutputUnits(u32 port) const;
	virtual void updateNumberOfUnits(u32 port);

	//virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);

};
}//namespace
#endif /* SRC_NN_MULTIPORTLAYER_HH_ */
