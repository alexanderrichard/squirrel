#ifndef NN_NEURALNETWORK_HH_
#define NN_NEURALNETWORK_HH_

#include <vector>
#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Layer.hh"
#include "Connection.hh"

namespace Nn {

class NeuralNetwork
{
private:
	static const Core::ParameterStringList paramConnections_;
	static const Core::ParameterInt paramInputDimension_;
private:
	u32 inputDimension_;
	u32 nLayer_;
	InputLayer inputLayer_;							// connects input to the network
	std::vector<Layer*> layer_;
	std::vector<Connection*> connections_;
	std::map<std::string, u32> layerNameToIndex_;

	bool isRecurrent_;
	bool isInitialized_;
	bool isComputing_;

	bool layerExists(std::string& layerName);
	void addLayer(std::string& layerName);
	void addConnection(Connection* connection, bool needsWeightsFileSuffix);
	void buildNetwork();
	bool containsLayer(BaseLayer* layer, std::vector<BaseLayer*> v);
	void sortTopologically();
	void logTopology();
public:
	NeuralNetwork();
	virtual ~NeuralNetwork();
	/*
	 * @param minibatchSize the size of a mini batch
	 * @param maxMemory the number of time frames memorized in the activations/error signal container
	 */
	void initialize(u32 maxMemory = 1);

	/*
	 * add a connection to the network
	 * @param connectionName the name of the connection
	 * @param sourceLayer, destLayer the source and destination of the connection
	 * @param sourcePort, destPort the ports to tie the connection to in source and destination layer
	 * @param needsWeigthsFileSuffix if true, the port numbers will be added to the weights filename automatically
	 */
	// public because it is used in Layer::addInternalConnections
	void addConnection(std::string& connectionName, BaseLayer* sourceLayer, BaseLayer* destLayer,
			u32 sourcePort, u32 destPort, bool needsWeightsFileSuffix = false);

	bool isRecurrent() const { return isRecurrent_; }
	// index of the last recurrent layer (0 if no such layer exists)
	u32 lastRecurrentLayerIndex() const;

	/*
	 * @param batchSize the number of feature vectors in a minibatch
	 */
	void setMinibatchSize(u32 batchSize);
	/*
	 * @param maxMemory the maximal amount of timeframes that are memorized for recurrent neural networks
	 */
	void setMaximalMemory(u32 maxMemory);

	u32 nLayer() const { return nLayer_; }
	u32 nConnections() const { return connections_.size(); }
	/*
	 * @return reference to the layer at the given index
	 */
	Layer& layer(u32 layerIndex) const;
	/*
	 * @return reference to the last layer in the network
	 */
	Layer& outputLayer() const { return layer(nLayer_ - 1); }
	/*
	 * @return reference to the connection with the given index
	 */
	Connection& connection(u32 connectionIndex) const;

	u32 inputDimension() const { return inputDimension_; }
	u32 outputDimension() const { return outputLayer().nUnits(); }

	// forward the given mini batch up to layer maxLayerIndex (default: forward to the output layer)
	void forward(Matrix& minibatch, u32 maxLayerIndex = Types::max<u32>());
	void forwardTimeframe(MatrixContainer& batchedSequence, u32 t, bool greedyForwarding = true);
	void forwardSequence(MatrixContainer& batchedSequence);

	void reset();

	// in training mode, backpropagation is done
	void setTrainingMode(bool trainingMode, u32 firstTrainableLayerIndex = 0);

	// save the network parameters (optionally with a suffix for each weight and bias file)
	void saveNeuralNetworkParameters();
	void saveNeuralNetworkParameters(const std::string& suffix);

	bool isComputing() const { return isComputing_; }
	void initComputation(bool sync = true);
	void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_NEURALNETWORK_HH_ */
