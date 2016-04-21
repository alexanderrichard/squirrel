#ifndef NN_CONNECTION_HH_
#define NN_CONNECTION_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"

namespace Nn {

// forward declaration of BaseLayer
class BaseLayer;

/*
 * base class for neural network connections
 */
class Connection
{
private:
	static const Core::ParameterEnum paramConnectionType_;
	static const Core::ParameterFloat paramWeightScale_;
public:
	enum ConnectionType { plainConnection, weightConnection };
protected:
	std::string name_;			// the name of the connection
	std::string prefix_;		// config prefix for the connection (neural-network.<connection-name>)
	BaseLayer* source_;
	BaseLayer* dest_;
	u32 sourcePort_;
	u32 destPort_;
	bool isComputing_;
	ConnectionType connectionType_;
	std::string weightsFileSuffix_;
	Float weightScale_; 		// scale the weights with this value, e.g. if dropout is used

	Matrix dummyWeights_;		// empty matrix, just a dummy
private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);
public:
	Connection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, ConnectionType type);
	virtual ~Connection() {}
	const std::string& name() const { return name_; }
	const ConnectionType type() const { return connectionType_; }
	virtual void initialize();
	virtual bool hasWeights() const { return false; }
	virtual bool isTrainable() const { return false; }
	virtual bool isRecurrent() const;
	/*
	 * return references to the source and target layer of this connections (and the respective port numbers)
	 */
	virtual BaseLayer& from();
	virtual BaseLayer& to();
	virtual u32 sourcePort() { return sourcePort_; }
	virtual u32 destinationPort() { return destPort_; }

	// this is not possible for connections without weights
	virtual Matrix& weights() { require(hasWeights()); return dummyWeights_; }

	// in forward pass, multiply with transposed weight matrix, in backpropagation multiply with non-transposed weigth matrix
	virtual void forwardWeightMultiplication();
	virtual void backpropagateWeights(u32 timeframe = 0);

	// save weights to a file
	virtual void saveWeights(const std::string& suffix) {};
	// set a suffix for the weights file
	virtual void setWeightsFileSuffix();

	virtual bool isComputing() const { return isComputing_; }
	virtual void initComputation(bool sync = true) { isComputing_ = true; }
	virtual void finishComputation(bool sync = true) { isComputing_ = false; }

	/* factory */
	static Connection* createConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort);
};

/*
 * neural network connection with weight matrix
 */
class WeightConnection : public Connection
{
private:
	typedef Connection Precursor;
protected:
	static const Core::ParameterBool paramIsTrainable_;
	static const Core::ParameterEnum paramWeightInitialization_;
	static const Core::ParameterFloat paramRandomWeightMin_;
	static const Core::ParameterFloat paramRandomWeightMax_;
	static const Core::ParameterString paramOldWeightsFilename_;
	static const Core::ParameterString paramNewWeightsFilename_;
	enum WeightInitialization { random, zero, identity };
protected:
	std::string oldWeightsFile_;
	std::string newWeightsFile_;
	Matrix weights_;
	bool isTrainable_;
	virtual void _initializeWeights(u32 nRows, u32 nColumns);
	virtual void initializeWeights();
private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);
public:
	WeightConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, ConnectionType type);
	virtual ~WeightConnection() {}
	virtual void initialize();
	virtual bool hasWeights() const { return true; }
	virtual bool isTrainable() const;
	virtual Matrix& weights();
public:
	// weights are (optionally) loaded from an old weights file and written to a new weights file
	virtual void setOldWeightsFile(const std::string& filename);
	virtual void setNewWeightsFile(const std::string& filename);
	virtual void saveWeights(const std::string& suffix);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_CONNECTION_HH_ */
