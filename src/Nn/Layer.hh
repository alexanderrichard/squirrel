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
 * Layer.hh
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#ifndef NN_LAYER_HH_
#define NN_LAYER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Connection.hh"
#include "MatrixContainer.hh"

namespace Nn {

// forward declaration of Nn::NeuralNetwork
class NeuralNetwork;

/*
 * base class for all neural network layer
 */
class BaseLayer
{
private:
	static const Core::ParameterInt paramNumberOfUnits_;
	static const Core::ParameterInt paramNumberOfPorts_;
	// dropout is only used if dropout probability > 0
	static const Core::ParameterFloat paramDropoutProbability_;

protected:
	std::string name_;			// name of the layer
	std::string prefix_;		// config prefix for the layer (neural-network.<layer-name>)
	u32 nUnits_;				// number of layer units
	u32 width_;					// used for CNNs (input width)
	u32 height_;				// used for CNNs (input height)
	u32 nChannels_;				// used for CNNs (input channels)
	u32 nPorts_;				// number of ports

	Float dropoutProbability_;
	bool useDropout_;

protected:
	u32 nTimeframes_;			// number of timeframes

	std::vector<MatrixContainer> activations_;	// the activations of the layer, one container for each port
	std::vector<MatrixContainer> errorSignals_;	// the error signals of the layer (only used in training), one container for each port
	std::vector<MatrixContainer> dropoutMasks_;  // store binary masks if dropout is used

	bool trainingMode_;
	bool isInitialized_;
	bool isComputing_;			// are the layer matrices/vectors in computing state?
public:
	BaseLayer(const char* name);
	virtual ~BaseLayer() {}
	virtual void initialize(const std::string &basePath, const std::string& suffix, u32 maxMemory = 1);

	virtual const std::string& name() const { return name_; }

	// nInputUnits and nOutputUnits is usually the same if not specified differently in the respective layer
	virtual u32 nInputUnits(u32 port) const { return nUnits_; } // nInputUnits can depend on the port in special cases
	virtual u32 nOutputUnits(u32 port) const { return nUnits_; } // nOutputUnits can depend on the port in special cases
	// needed for CNNs
	virtual u32 width(u32 port) const { return width_; }
	virtual u32 height(u32 port) const { return height_; }
	virtual u32 nChannels(u32 port) const { return nChannels_; }

	// returns true if all timeframes need to be forwarded/backpropagated before proceeding to the next layer is possible
	// this is necessary for some recurrent networks, e.g. for bidirectional networks in the TemporalReversionLayer
	virtual bool requiresFullPass() const { return false; }

	// port is not used for standard layers but may be important if input dimension is different at different input ports
	// (e.g. in some MultiPortLayers such as the AttentionLayer)
	virtual void setWidth(u32 port, u32 width);
	virtual void setHeight(u32 port, u32 height);
	virtual void setNumberOfChannels(u32 port, u32 channels);
	virtual void updateNumberOfUnits(u32 port);

	virtual u32 nInputPorts() const { return nPorts_; }
	virtual u32 nOutputPorts() const { return nPorts_; }

	virtual void setTrainingMode(bool trainingMode);
	virtual bool isInTrainingMode() const { return trainingMode_; }

	/* port blocking: prevent the network from establishing (other than internal) connections from/to blocked ports */
	virtual void blockPorts() {}
	virtual void unblockPorts() {}
	virtual bool isInputPortBlocked(u32 port) { return false; }
	virtual bool isOutputPortBlocked(u32 port) { return false; }

	/* activations */
	// the activations bound to the input ports
	virtual Matrix& activationsIn(u32 timeframe, u32 port);
	// the activations bound to the output ports (by default the same as the input activations)
	virtual Matrix& activationsOut(u32 timeframe, u32 port) { require_lt(port, nOutputPorts()); return activationsIn(timeframe, port); }
	// alias for activationsOut (name is more intuitive and usually we want to have the activations after application of the nonlinearity)
	virtual Matrix& activations(u32 timeframe, u32 port) { return activationsOut(timeframe, port); }
	virtual Matrix& latestActivations(u32 port); // return latest activation

	/* error signals */
	// the error signals bound to the input ports
	virtual Matrix& errorSignalIn(u32 timeframe, u32 port);
	// the error signals bound to the output ports
	virtual Matrix& errorSignalOut(u32 timeframe, u32 port) { require_lt(port, nOutputPorts()); return errorSignalIn(timeframe, port); }
	// alias for errorSignalIn (name is more intuitive and usually we want to have the error signal after application of the derived nonlinearity)
	virtual Matrix& errorSignal(u32 timeframe, u32 port) { return errorSignalIn(timeframe, port); }
	virtual Matrix& latestErrorSignal(u32 port); // return latest error signal of respective input port

	// adds a time frame to the activations and error signals container (interesting for recurrent neural networks)
	virtual void addTimeframe(u32 minibatchSize);
	// adds a time frame to the activations and error signals container (interesting for recurrent neural networks), no memory allocation
	virtual void addEmptyTimeframe();
	// resize activations and error signals
	virtual void resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns);
	virtual void setMaximalMemory(u32 maxMemory);
	// reset the containers (nTimeframes_ := 0)
	virtual void reset();
	// return the number of activation/error signal time frames
	virtual u32 nTimeframes() const { return nTimeframes_; }

	/* function to hide or unhide columns in the activations and error signal containers */
	virtual void setActivationVisibility(u32 timeframe, u32 nVisibleColumns);
	virtual void setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns);

	virtual bool isInitialized() const { return isInitialized_; }

	virtual bool useDropout() const { return useDropout_; }
	virtual Float dropoutProbability() const { return dropoutProbability_; }
	virtual void dropout();

	virtual bool isComputing() const { return isComputing_; }
	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * linear layer (base class for layers that can occur in neural networks)
 */
class Layer : public BaseLayer
{
private:
	typedef BaseLayer Precursor;
private:
	static const Core::ParameterEnum paramLayerType_;
private:
	// bias parameters
	static const Core::ParameterBool paramUseBias_;
	static const Core::ParameterBool paramIsBiasTrainable_;
	static const Core::ParameterEnum paramBiasInitialization_;
	static const Core::ParameterFloat paramRandomBiasMin_;
	static const Core::ParameterFloat paramRandomBiasMax_;
	static const Core::ParameterBool paramUseCudnn_;
	static const Core::ParameterFloat paramLearningRateFactor_;

public:
	enum LayerType { identity, sigmoid, tanh, softmax, max, exponential, logarithmic, rectified, triangle, clipped,
		l2normalization, powerNormalization, polynomialPreprocessing, fisherEncoding, sequenceLengthNormalization,
		featureCloning, approximateFeatureMap, modulatedSum, maxout, multiplication, temporalAveraging, concatenation, gatedRecurrentUnit,
		attention, maxPoolingLayer, avgPoolingLayer, batchNormalizationLayer, temporalReversion, preProcessing };
protected:
	enum ParamInitialization { random, zero };
	// layer type
	LayerType type_;
private:
	std::vector<u32> nIncomingConnections_;			// incoming connections per port
	std::vector<u32> nOutgoingConnections_;			// outgoing connections per port
	std::vector< std::vector<Connection*> > incomingConnections_;	// incoming connections per port
	std::vector< std::vector<Connection*> > outgoingConnections_;	// outgoing connections per port

protected:
	// bias
	std::vector<Vector> bias_;	// bias for each port

	bool useBias_;				// if false, bias is ignored
	bool isBiasTrainable_;

	bool isRecurrent_;			// has this layer a recurrent connection?
	bool isOutputLayer_;
	bool useCudnn_;
	f32 learningRateFactor_;

	virtual std::string getParamFileName(const std::string& basePath, const std::string& paramName,
			const std::string& suffix, u32 port);
	void _initializeBias(u32 port);
	void initializeBias(const std::string &basePath, const std::string& suffix);

	virtual void _initializeParam(Vector &param, ParamInitialization initMethod);
protected:
	virtual void forward(u32 port); // forward the given port
	virtual void backpropagate(u32 timeframe, u32 port);
public:
	Layer(const char* name);
	virtual ~Layer() {};
	/*
	 * @param maxMemory maximal number of time frames that are memorized (history of the activations for recurrent neural networks)
	 */
	virtual void initialize(const std::string &basePath, const std::string& suffix, u32 maxMemory = 1);
	// add internal connection, e.g. in case of complex recurrent units such as LSTMs or GRUs
	// called after creation of the layer in NeuralNetwork::addLayer
	virtual void addInternalConnections(NeuralNetwork& network) {}
	virtual void saveParams(const std::string& basePath,const std::string& suffix);
	void saveBias(const std::string& basePath, const std::string& suffix);

	/* connections */
	u32 nIncomingConnections(u32 port);
	u32 nOutgoingConnections(u32 port);
	virtual void addIncomingConnection(Connection* c, u32 port);
	virtual void addOutgoingConnection(Connection* c, u32 port);
	Connection& incomingConnection(u32 incomingConnectionIndex, u32 port);
	Connection& outgoingConnection(u32 outgoingConnectionIndex, u32 port);

	virtual bool isTrainable(u32 incomingConnectionIndex, u32 port) const;
	Matrix& weights(u32 incomingConnectionIndex, u32 port);
	Vector& bias(u32 port);
	virtual bool useBias() const { return useBias_; }
	virtual void setUseBias(bool useBias) { useBias_ = useBias; }
	virtual bool isBiasTrainable() const { return (useBias_ && isBiasTrainable_); }

	virtual LayerType layerType() const { return type_; }
	virtual bool isRecurrent() const { return isRecurrent_; }

	void setAsOutputLayer();
	bool isOutputLayer() { return isOutputLayer_; }

	/* layer processing (forwarding, backpropagation) */
	virtual void forward(); // compute the activations for the layer input at the current time frame
	virtual void finalizeForwarding() {}
	// backpropagate all ports
	virtual void backpropagate(u32 timeframe = 0);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);

	virtual f32 learningRateFactor() { return learningRateFactor_; }
	/* factory */
	static Layer* createLayer(const char* name);
};

/*
 * input layer (connects input to neural network)
 */
class InputLayer : public BaseLayer
{
private:
	typedef BaseLayer Precursor;
private:
	std::vector<Matrix*> input_;
public:
	InputLayer();
	virtual ~InputLayer() {}
	virtual void initialize(u32 inputDimension, u32 width, u32 height, u32 nChannels);

	virtual void setTrainingMode(bool trainingMode) {}
	virtual bool isInTrainingMode() const { return false; }

	// add the given input at the next position in the array (next time frame)
	void addInput(Matrix& input);

	virtual void resize(u32 nRows, u32 nColumns, u32 maxMemory = 1) {}
	virtual void reset();
	virtual Matrix& activationsOut(u32 timeframe, u32 port);
	virtual u32 nTimeframes() const { return input_.size(); }

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);

};

} // namespace

#endif /* NN_LAYER_HH_ */
