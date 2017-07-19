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
 * NeuralNetwork.hh
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

#ifndef NN_NEURALNETWORK_HH_
#define NN_NEURALNETWORK_HH_

#include <vector>
#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Layer.hh"
#include "Connection.hh"
#include "MultiPortLayer.hh"
namespace Nn {

class NeuralNetwork
{
private:
	static const Core::ParameterStringList paramConnections_;
	static const Core::ParameterInt paramInputDimension_;

	static const Core::ParameterInt paramSourceWidth_;
	static const Core::ParameterInt paramSourceHeight_;
	static const Core::ParameterInt paramSourceChannels_;

	static const Core::ParameterString paramWriteParamsTo_;
	static const Core::ParameterString paramLoadParamsFrom_;
	static const Core::ParameterInt paramLoadParamsEpoch_;
private:
	u32 inputDimension_;

	u32 sourceWidth_;
	u32 sourceHeight_;
	u32 sourceChannels_;


	u32 nLayer_;
	InputLayer inputLayer_;							// connects input to the network
	std::vector<Layer*> layer_;
	std::vector<Connection*> connections_;
	std::map<std::string, u32> layerNameToIndex_;

	bool isRecurrent_;
	bool isInitialized_;
	bool isComputing_;

	std::string writeParamsTo_;
	std::string loadParamsFrom_;
	u32 loadParamsEpoch_;

	bool layerExists(std::string& layerName);
	void addLayer(std::string& layerName);
	void addConnection(Connection* connection, bool needsWeightsFileSuffix);
	void buildNetwork();
	bool containsLayer(BaseLayer* layer, std::vector<BaseLayer*> v);
	void checkTopology();
	void logTopology();
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();
	/*
	 * @param minibatchSize the size of a mini batch
	 * @param maxMemory the number of time frames memorized in the activations/error signal container
	 */
	void initialize(u32 maxMemory = 1);

	// public because it is used in Layer::addInternalConnections
	void addConnection(std::string& connectionName, BaseLayer* sourceLayer, BaseLayer* destLayer,
			u32 sourcePort, u32 destPort, bool needsWeightsFileSuffix = false);

	bool isRecurrent() const { return isRecurrent_; }
	bool requiresFullMemorization() const; // do all timeframes need to be stored for sequence forwarding?
	u32 lastRecurrentLayerIndex() const;

	void setMinibatchSize(u32 batchSize);
	void setMaximalMemory(u32 maxMemory);
	u32 nLayer() const { return nLayer_; }
	u32 nConnections() const { return connections_.size(); }
	Layer& layer(u32 layerIndex) const;
	Layer& layer(std::string& name)  const;
	Layer& outputLayer() const { return layer(nLayer_ - 1); }
	Connection& connection(u32 connectionIndex) const;

	u32 inputDimension() const { return inputDimension_; }
	u32 outputDimension() const { return outputLayer().nOutputUnits(0); }

	// forward the given mini batch up to layer maxLayerIndex (default: forward to the output layer)
	void forward(Matrix& minibatch, u32 layerIndexFrom = 0, u32 layerIndexTo = Types::max<u32>());
	void forwardTimeframe(MatrixContainer& batchedSequence, u32 t, u32 layerIndexFrom = 0, u32 layerIndexTo = Types::max<u32>(), bool greedyForwarding = true);
	void forwardSequence(MatrixContainer& batchedSequence, bool greedyForwarding = true);

	void reset();

	void setTrainingMode(bool trainingMode, u32 firstTrainableLayerIndex = 0);

	void saveNeuralNetworkParameters();
	void saveNeuralNetworkParameters(const std::string& suffix);

	bool isComputing() const { return isComputing_; }
	void initComputation(bool sync = true);
	void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_NEURALNETWORK_HH_ */
