#ifndef NN_REGULARIZER_HH_
#define NN_REGULARIZER_HH_

#include <Core/CommonHeaders.hh>
#include "NeuralNetwork.hh"
#include "Statistics.hh"

namespace Nn {

/*
 * Base class for regularizers
 */
class Regularizer
{
private:
	static const Core::ParameterEnum paramRegularizerType_;
	enum RegularizerType { none, l2Regularizer };
	static const Core::ParameterFloat paramRegularizationConstant_;
protected:
	Float regularizationConstant_;
public:
	Regularizer();
	virtual ~Regularizer() {}
	/*
	 * @param network the neural network instance
	 * @param statistics the statistics object containing the gradient and the objective function value
	 * includes regularizer into objective function / gradient
	 */
	virtual void addToObjectiveFunction(NeuralNetwork& neuralNetwork, Statistics& statistics) {}
	virtual void addToGradient(NeuralNetwork& neuralNetwork, Statistics& statistics) {}

	/* factory */
	static Regularizer* createRegularizer();
};

/*
 * l2-regularizer
 * regularizationConstant_ / 2 * ||Parameters||^2
 */
class L2Regularizer : public Regularizer
{
private:
	typedef Regularizer Precursor;
public:
	L2Regularizer();
	virtual ~L2Regularizer() {}
	virtual void addToObjectiveFunction(NeuralNetwork& network, Statistics& statistics);
	virtual void addToGradient(NeuralNetwork& network, Statistics& statistics);
};

} // namespace

#endif /* NN_REGULARIZER_HH_ */
