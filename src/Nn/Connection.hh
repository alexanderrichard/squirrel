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
 * Connection.hh
 *
 *  Created on: May 13, 2014
 *      Author: richard
 */

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
	static const Core::ParameterFloat paramLearningRateFactor_;
	static const Core::ParameterBool paramIsRecurrent_;

public:
	enum ConnectionType { plainConnection, weightConnection, convolutionalConnection, validConvolutionalConnection };
protected:
	std::string name_;			// the name of the connection
	std::string prefix_;		// config prefix for the connection (neural-network.<connection-name>)
	BaseLayer* source_;
	BaseLayer* dest_;
	u32 sourcePort_;
	u32 destPort_;
	bool isRecurrent_;			// a connection is recurrent if the source layer index is greater or equal to the target layer index
	bool isComputing_;
	ConnectionType connectionType_;
	std::string weightsFileSuffix_;

	Matrix dummyWeights_;		// empty matrix, just a dummy
	f32 learningRateFactor_;
	virtual std::string getParamFileName(const std::string& basePath, const std::string& suffix);
private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);
public:
	Connection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type);
	virtual ~Connection() {}
	const std::string& name() const { return name_; }
	const ConnectionType type() const { return connectionType_; }
	virtual void initialize() {}
	virtual void initializeWeights(const std::string& basePath, const std::string& suffix) {}
	virtual bool hasWeights() const { return false; }
	virtual bool isTrainable() const { return false; }
	virtual bool isRecurrent() const;
	virtual BaseLayer& from();
	virtual BaseLayer& to();
	virtual u32 sourcePort() { return sourcePort_; }
	virtual u32 destinationPort() { return destPort_; }

	// this is not possible for connections without weights
	virtual Matrix& weights() { require(hasWeights()); return dummyWeights_; }

	virtual void forwardWeightMultiplication();
	virtual void backpropagateWeights(u32 timeframe = 0);

	virtual void saveWeights(const std::string& basePath, const std::string& suffix) {};
	virtual void setWeightsFileSuffix();

	virtual bool isComputing() const { return isComputing_; }
	virtual void initComputation(bool sync = true) { isComputing_ = true; }
	virtual void finishComputation(bool sync = true) { isComputing_ = false; }

	virtual f32 learningRateFactor() { return learningRateFactor_; }

	/* factory */
	static Connection* createConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent);
};

/*
 * Plain Connection
 */
class PlainConnection : public Connection
{
private:
	typedef Connection Precursor;
public:
	PlainConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type);
	virtual ~PlainConnection() {}
	virtual void initialize();
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

	enum WeightInitialization { random, zero, identity, glorot };
protected:
	Matrix weights_;
	bool isTrainable_;

	virtual void _initializeWeights(u32 nRows, u32 nColumns);
	virtual void _initializeWeights(const std::string& basePath, const std::string& suffix, u32 nRows, u32 nColumns);
private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);
public:
	WeightConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type);
	virtual ~WeightConnection() {}
	virtual void initialize();
	virtual void initializeWeights(const std::string& basePath, const std::string& suffix);
	virtual bool hasWeights() const { return true; }
	virtual bool isTrainable() const;
	virtual Matrix& weights();
public:
	virtual void saveWeights(const std::string& basePath, const std::string& suffix);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * ConvolutionalConnection with convolutional Kernel
 */
class ConvolutionalConnection : public WeightConnection
{
private:
	typedef WeightConnection Precursor;

protected:

	static const Core::ParameterInt paramKernelWidth_;
	static const Core::ParameterInt paramKernelHeight_;
	static const Core::ParameterInt paramDestChannels_;
	//static const Core::ParameterString paramConvolutionType_;
	static const Core::ParameterInt paramStrideX_;
	static const Core::ParameterInt paramStrideY_;

	u32 kernelHeight_;
	u32 kernelWidth_;
	u32 destChannels_;
	u32 strideX_;
	u32 strideY_;

#ifdef MODULE_CUDNN
	u32 previousBatchSize_;
	CudnnConvolution cudnnConvolution;
#endif

private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);

protected:
	virtual void forward(const Matrix& source, Matrix& dest);
	virtual void backward(const Matrix& source, Matrix& dest);

public:
	ConvolutionalConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type);
	virtual ~ConvolutionalConnection();

	virtual void initialize();
	virtual void initializeWeights(const std::string& basePath, const std::string& suffix);

	virtual u32 getResultWidth(u32 sourceWidth, u32 kernelWidth, u32 strideX);
	virtual u32 getResultHeight(u32 sourceHeight, u32 kernelHeight, u32 strideY);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);

	u32 kernelWidth();
	u32 kernelHeight();
	u32 destChannels();

	virtual void forwardPreprocess(const Matrix& source, Matrix& dest);
	virtual void forwardPostProcess(const Matrix& source, Matrix& dest, u32 sourceColumns);

	virtual void backwardPreprocess(const Matrix& source, Matrix& dest);
	virtual void backwardPostprocess(const Matrix& source, Matrix& dest, u32 destColumns);

	virtual void backwardWRTKernel(Matrix &weightsGradient, const Matrix &activationIn, const Matrix &errorSignalOut);
};

class ValidConvolutionalConnection: public ConvolutionalConnection {
private:
	typedef ConvolutionalConnection Precursor;
private:
	virtual void _forwardWeightMultiplication(const Matrix& source, Matrix& dest);
	virtual void _backpropagateWeights(const Matrix& source, Matrix& dest);
public:
	ValidConvolutionalConnection(const char* name, BaseLayer* source, BaseLayer* dest, u32 sourcePort, u32 destPort, bool isRecurrent, ConnectionType type);
	virtual ~ValidConvolutionalConnection();

	virtual u32 getResultWidth(u32 sourceWidth, u32 kernelWidth, u32 strideX);
	virtual u32 getResultHeight(u32 sourceHeight, u32 kernelHeight, u32 strideY);

	virtual void forwardPreprocess(const Matrix& source, Matrix& dest);
	virtual void backwardPostprocess(const Matrix& source, Matrix& dest, u32 destColumns);
};

} // namespace

#endif /* NN_CONNECTION_HH_ */
