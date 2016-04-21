#include "Estimator.hh"

using namespace Nn;
using namespace std;

/*
 * Estimator
 */
const Core::ParameterEnum Estimator::paramEstimatorType_("method", "none, steepest-descent, rprop", "none", "estimator");

const Core::ParameterBool Estimator::paramLogParameterNorm_("log-parameter-norm", true, "estimator");

const Core::ParameterBool Estimator::paramLogGradientNorm_("log-gradient-norm", true, "estimator");

const Core::ParameterBool Estimator::paramLogStepSize_("log-step-size", true, "estimator");

Estimator::Estimator() :
		epoch_(1),
		learningRateSchedule_(0),
		logParameterNorm_(Core::Configuration::config(paramLogParameterNorm_)),
		logGradientNorm_(Core::Configuration::config(paramLogGradientNorm_)),
		logStepSize_(Core::Configuration::config(paramLogStepSize_))
{}

void Estimator::initialize(NeuralNetwork& network) {
	learningRateSchedule_ = LearningRateSchedule::createLearningRateSchedule();
	learningRateSchedule_->initialize();
}

void Estimator::setEpoch(u32 epoch) {
	epoch_ = epoch;
}

void Estimator::estimate(NeuralNetwork& network, Statistics& statistics) {
	require(learningRateSchedule_);

	// compute the learning rate for the weight update
	learningRateSchedule_->updateLearningRate(statistics, epoch_);

	// log the parameter norm
	if (logParameterNorm_ && statistics.needsGradient()) {
		Float parameterNorm = 0.0;
		for (u32 c = 0; c < network.nConnections(); c++) {
			if (network.connection(c).hasWeights() && network.connection(c).isTrainable())
				parameterNorm += network.connection(c).weights().l1norm();
		}
		for (u32 l = 0; l < network.nLayer(); l++) {
			if (network.layer(l).useBias() && network.layer(l).isBiasTrainable()) {
				for (u32 port = 0; port < network.layer(l).nInputPorts(); port++)
					parameterNorm += network.layer(l).bias(port).l1norm();
			}
		}
		Core::Log::os("parameter-norm (l1-norm of all trainable weights and biases): ") << parameterNorm;
	}

	// log the gradient norm
	if (logGradientNorm_ && statistics.needsGradient()) {
		Core::Log::os("gradient-norm: ") << statistics.gradientNorm();
	}
}

Estimator* Estimator::createEstimator() {
	Estimator* estimator = 0;
	switch ( Core::Configuration::config(paramEstimatorType_) ) {
	case steepestDescent:
		Core::Log::os("Create steepest-descent estimator.");
		estimator = new SteepestDescentEstimator();
		break;
	case rprop:
		Core::Log::os("Create rprop estimator.");
		estimator = new RpropEstimator();
		break;
	case none:
		estimator = new Estimator();
		break;
	default:
		break;
	}
	require(estimator);
	return estimator;
}

/*
 * SteepestDescentEstimator
 */
const Core::ParameterFloat SteepestDescentEstimator::paramBiasWeight_("bias-weight", 1.0, "estimator");

SteepestDescentEstimator::SteepestDescentEstimator() :
		Precursor(),
		biasWeight_(Core::Configuration::config(paramBiasWeight_))
{}

void SteepestDescentEstimator::estimate(NeuralNetwork& network, Statistics& statistics) {
	Precursor::estimate(network, statistics);
	Float learningRate = learningRateSchedule_->learningRate();
	Float stepSize = 0;
	// update bias
	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).useBias() && network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				network.layer(l).bias(port).add(statistics.biasGradient(network.layer(l).name(), port), -learningRate * biasWeight_);
				if (logStepSize_)
					stepSize += statistics.biasGradient(network.layer(l).name(), port).l1norm() * learningRate * biasWeight_;
			}
		}
	}
	// update weights
	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			network.connection(c).weights().add(statistics.weightsGradient(network.connection(c).name()), -learningRate);
			if (logStepSize_)
				stepSize += statistics.weightsGradient(network.connection(c).name()).l1norm() * learningRate;
		}
	}
	// log step size
	if (logStepSize_)
		Core::Log::os("step size: ") << stepSize;
}

/*
 * RpropEstimator
 */
const Core::ParameterFloat RpropEstimator::paramIncreasingFactor_("increasing-factor", 1.2, "estimator");
const Core::ParameterFloat RpropEstimator::paramDecreasingFactor_("decreasing-factor", 0.5, "estimator");
const Core::ParameterFloat RpropEstimator::paramMaxUpdateValue_("max-update-value", 50.0, "estimator");
const Core::ParameterFloat RpropEstimator::paramMinUpdateValue_("min-update-value", 0.000001, "estimator");
const Core::ParameterFloat RpropEstimator::paramInitialStepSize_("initial-step-size", 0.000001, "estimator");
const Core::ParameterString RpropEstimator::paramLoadStepSizesFrom_("load-step-sizes-from", "", "estimator");
const Core::ParameterString RpropEstimator::paramWriteStepSizesTo_("write-step-sizes-to", "", "estimator");

RpropEstimator::RpropEstimator() :
		Precursor(),
		increasingFactor_(Core::Configuration::config(paramIncreasingFactor_)),
		decreasingFactor_(Core::Configuration::config(paramDecreasingFactor_)),
		maxUpdateValue_(Core::Configuration::config(paramMaxUpdateValue_)),
		minUpdateValue_(Core::Configuration::config(paramMinUpdateValue_)),
		initialStepSize_(Core::Configuration::config(paramInitialStepSize_)),
		oldGradients_((Statistics::StatisticTypes) (Statistics::baseStatistics | Statistics::gradientStatistics) ),
		updateValues_((Statistics::StatisticTypes) (Statistics::baseStatistics | Statistics::gradientStatistics) )
{}

RpropEstimator::~RpropEstimator() {
	std::string filename(Core::Configuration::config(paramWriteStepSizesTo_));
	if (!filename.empty()) {
		updateValues_.finishComputation();
		updateValues_.saveStatistics(filename);
	}
}

void RpropEstimator::initialize(NeuralNetwork& network) {
	Precursor::initialize(network);
	oldGradients_.initialize(network);
	updateValues_.initialize(network);

	oldGradients_.initComputation();
	updateValues_.initComputation();

	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).useBias() && network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				updateValues_.biasGradient(network.layer(l).name(), port).fill(initialStepSize_);
				oldGradients_.biasGradient(network.layer(l).name(), port).setToZero();
			}
		}
	}

	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			updateValues_.weightsGradient(network.connection(c).name()).fill(initialStepSize_);
			oldGradients_.weightsGradient(network.connection(c).name()).setToZero();
		}
	}

	std::string filename = Core::Configuration::config(paramLoadStepSizesFrom_);
	if (!filename.empty()) {
		updateValues_.finishComputation(false);
		updateValues_.loadStatistics(filename);
		updateValues_.initComputation();
	}

	Core::Log::os("Use RProp with initial-step-size ") << initialStepSize_;
}

void RpropEstimator::estimate(NeuralNetwork& network, Statistics& statistics) {
	Precursor::estimate(network, statistics);
	Float stepSize = 0;

	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).useBias() && network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				network.layer(l).bias(port).rpropUpdate(
						statistics.biasGradient(network.layer(l).name(), port),
						oldGradients_.biasGradient(network.layer(l).name(), port),
						updateValues_.biasGradient(network.layer(l).name(), port),
						increasingFactor_, decreasingFactor_, maxUpdateValue_, minUpdateValue_);
				if (logStepSize_)
					stepSize += updateValues_.biasGradient(network.layer(l).name(), port).l1norm();
			}
		}
	}

	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			network.connection(c).weights().rpropUpdate(
					statistics.weightsGradient(network.connection(c).name()),
					oldGradients_.weightsGradient(network.connection(c).name()),
					updateValues_.weightsGradient(network.connection(c).name()),
					increasingFactor_, decreasingFactor_, maxUpdateValue_, minUpdateValue_);
			if (logStepSize_)
				stepSize += updateValues_.weightsGradient(network.connection(c).name()).l1norm();
		}
	}
	// log step size
	if (logStepSize_)
		Core::Log::os("step size: ") << stepSize;
}

