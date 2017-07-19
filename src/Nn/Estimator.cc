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
 * Estimator.cc
 *
 *  Created on: May 21, 2014
 *      Author: richard
 */

#include "Estimator.hh"

using namespace Nn;
using namespace std;

/*
 * Estimator
 */
const Core::ParameterEnum Estimator::paramEstimatorType_("method", "none, steepest-descent, rprop, adam", "none", "estimator");

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

void Estimator::initialize(NeuralNetwork& network, TrainingTask task) {
	learningRateSchedule_ = LearningRateSchedule::createLearningRateSchedule();
	learningRateSchedule_->initialize(task);
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
			if (network.layer(l).isBiasTrainable()) {
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

void Estimator::finalize() {
	learningRateSchedule_->finalize();
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
	case adam:
		Core::Log::os("Create Adam estimator.");
		estimator = new AdamEstimator();
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

const Core::ParameterBool SteepestDescentEstimator::paramUseMomentum_("use-momentum", false, "estimator");

const Core::ParameterFloat SteepestDescentEstimator::paramMomentum_("momentum", 0.5f, "estimator");

SteepestDescentEstimator::SteepestDescentEstimator() :
		Precursor(),
		biasWeight_(Core::Configuration::config(paramBiasWeight_)),
		useMomentum_(Core::Configuration::config(paramUseMomentum_)),
		momentum_(Core::Configuration::config(paramMomentum_)),
		momentumStats_(Statistics::gradientStatistics)
{}

void SteepestDescentEstimator::initialize(NeuralNetwork& network, TrainingTask task) {
	Precursor::initialize(network, task);
	momentumStats_.initialize(network);
	momentumStats_.initComputation();
}

void SteepestDescentEstimator::estimate(NeuralNetwork& network, Statistics& statistics) {
	Precursor::estimate(network, statistics);
	Float learningRate = learningRateSchedule_->learningRate();
	Float stepSize = 0;
	// update bias
	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				if (useMomentum_) {
					momentumStats_.biasGradient(network.layer(l).name(), port).scale(momentum_);
					momentumStats_.biasGradient(network.layer(l).name(), port).add(statistics.biasGradient(network.layer(l).name(), port),
							-learningRate * biasWeight_ * network.layer(l).learningRateFactor());
					network.layer(l).bias(port).add(momentumStats_.biasGradient(network.layer(l).name(), port));
					if (logStepSize_) {
						stepSize += momentumStats_.biasGradient(network.layer(l).name(), port).l1norm();
					}
				}
				else {
					network.layer(l).bias(port).add(statistics.biasGradient(network.layer(l).name(), port), -learningRate * biasWeight_ * network.layer(l).learningRateFactor());
					if (logStepSize_)
						stepSize += statistics.biasGradient(network.layer(l).name(), port).l1norm() * learningRate * biasWeight_ * network.layer(l).learningRateFactor();
				}
			}
		}
	}
	// update weights
	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			if (useMomentum_) {
				momentumStats_.weightsGradient(network.connection(c).name()).scale(momentum_);
				momentumStats_.weightsGradient(network.connection(c).name()).add(statistics.weightsGradient(network.connection(c).name()),
						-learningRate * network.connection(c).learningRateFactor());
				network.connection(c).weights().add(momentumStats_.weightsGradient(network.connection(c).name()));
				if (logStepSize_) {
					stepSize += momentumStats_.weightsGradient(network.connection(c).name()).l1norm();
				}
			}
			else {
				network.connection(c).weights().add(statistics.weightsGradient(network.connection(c).name()), -learningRate * network.connection(c).learningRateFactor());
				if (logStepSize_)
					stepSize += statistics.weightsGradient(network.connection(c).name()).l1norm() * learningRate * network.connection(c).learningRateFactor();
			}
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
		oldGradients_((Statistics::StatisticTypes) (Statistics::gradientStatistics) ),
		updateValues_((Statistics::StatisticTypes) (Statistics::gradientStatistics) )
{}

RpropEstimator::~RpropEstimator() {
	std::string filename(Core::Configuration::config(paramWriteStepSizesTo_));
	if (!filename.empty()) {
		updateValues_.finishComputation();
		updateValues_.saveStatistics(filename);
	}
}

void RpropEstimator::initialize(NeuralNetwork& network, TrainingTask task) {
	Precursor::initialize(network, task);
	oldGradients_.initialize(network);
	updateValues_.initialize(network);

	oldGradients_.initComputation();
	updateValues_.initComputation();

	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).isBiasTrainable()) {
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
		if (network.layer(l).isBiasTrainable()) {
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
/*
 * ADAM Estimator
 */
const Core::ParameterFloat AdamEstimator::paramBeta1_("beta1", 0.9, "estimator");
const Core::ParameterFloat AdamEstimator::paramBeta2_("beta2", 0.999, "estimator");
const Core::ParameterFloat AdamEstimator::paramEpsilon_("epsilon", 0.00000001, "estimator");

AdamEstimator::AdamEstimator() :
		Precursor(),
		beta1_(Core::Configuration::config(paramBeta1_)),
		beta2_(Core::Configuration::config(paramBeta2_)),
		epsilon_(Core::Configuration::config(paramEpsilon_)),
		iteration_(0),
		firstMoment_((Statistics::StatisticTypes)(Statistics::gradientStatistics)),
		secondMoment_((Statistics::StatisticTypes)(Statistics::gradientStatistics)){

}
void AdamEstimator::initialize(NeuralNetwork& network, TrainingTask task) {

	Precursor::initialize(network, task);
	firstMoment_.initialize(network);
	secondMoment_.initialize(network);

	firstMoment_.initComputation();
	secondMoment_.initComputation();

	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				firstMoment_.biasGradient(network.layer(l).name(), port).setToZero();
				secondMoment_.biasGradient(network.layer(l).name(), port).setToZero();
			}
		}
	}
	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			firstMoment_.weightsGradient(network.connection(c).name()).setToZero();
			secondMoment_.weightsGradient(network.connection(c).name()).setToZero();
		}
	}
}

void AdamEstimator::update(Matrix &weights, const Matrix &firstMoment, const Matrix &secondMoment, Float learningRateFactor) {
	Matrix biasCorrectedFirstMoment;
	Matrix biasCorrectedSecondMoment;

	biasCorrectedFirstMoment.resize(firstMoment.nRows(), firstMoment.nColumns());
	biasCorrectedSecondMoment.resize(secondMoment.nRows(), secondMoment.nColumns());

	biasCorrectedFirstMoment.initComputation(false);
	biasCorrectedSecondMoment.initComputation(false);

	biasCorrectedFirstMoment.copy(firstMoment);
	biasCorrectedSecondMoment.copy(secondMoment);
	biasCorrectedFirstMoment.scale( (Float)  ( 1.0 / ( 1.0 - std::pow(beta1_, iteration_))));
	biasCorrectedSecondMoment.scale( (Float) ( 1.0 / ( 1.0 - std::pow(beta2_, iteration_))));

	biasCorrectedSecondMoment.signedPow((Float) 0.5);
	biasCorrectedSecondMoment.addConstantElementwise(epsilon_);
	biasCorrectedFirstMoment.elementwiseDivision(biasCorrectedSecondMoment);

	weights.add(biasCorrectedFirstMoment, (Float)(-1.0 * learningRate() * learningRateFactor));

	biasCorrectedFirstMoment.finishComputation(false);
	biasCorrectedSecondMoment.finishComputation(false);
}

void AdamEstimator::update(Vector &bias, const Vector &firstMoment, const Vector &secondMoment, Float learningRateFactor) {

	Vector biasCorrectedFirstMoment;
	Vector biasCorrectedSecondMoment;

	biasCorrectedFirstMoment.resize(firstMoment.nRows());
	biasCorrectedSecondMoment.resize(secondMoment.nRows());

	biasCorrectedFirstMoment.initComputation(false);
	biasCorrectedSecondMoment.initComputation(false);

	biasCorrectedFirstMoment.copy(firstMoment);
	biasCorrectedSecondMoment.copy(secondMoment);
	biasCorrectedFirstMoment.scale( (Float)  ( 1.0 / ( 1.0 - std::pow(beta1_, iteration_))));
	biasCorrectedSecondMoment.scale( (Float) ( 1.0 / ( 1.0 - std::pow(beta2_, iteration_))));

	biasCorrectedSecondMoment.signedPow((Float) 0.5);
	biasCorrectedSecondMoment.addConstantElementwise(epsilon_);
	biasCorrectedFirstMoment.elementwiseDivision(biasCorrectedSecondMoment);

	bias.add(biasCorrectedFirstMoment, (Float)(-1.0 * learningRate() * learningRateFactor));

	biasCorrectedFirstMoment.finishComputation(false);
	biasCorrectedSecondMoment.finishComputation(false);
}

void AdamEstimator::estimate(NeuralNetwork& network, Statistics& statistics) {
	Precursor::estimate(network, statistics);

	iteration_++;
	for (u32 l = 0; l < network.nLayer(); l++) {
		if (network.layer(l).isBiasTrainable()) {
			for (u32 port = 0; port < network.layer(l).nInputPorts(); port++) {
				firstMoment_.biasGradient(network.layer(l).name(), port).scale(beta1_);
				firstMoment_.biasGradient(network.layer(l).name(), port).add(statistics.biasGradient(network.layer(l).name(), port), (Float)(1.0 - beta1_));

				secondMoment_.biasGradient(network.layer(l).name(), port).scale(beta2_);
				statistics.biasGradient(network.layer(l).name(), port).elementwiseMultiplication(statistics.biasGradient(network.layer(l).name(), port));
				secondMoment_.biasGradient(network.layer(l).name(), port).add(statistics.biasGradient(network.layer(l).name(), port), (Float)(1.0 - beta2_));

				//if(network.layer(l).getLearningRate() != -1)
				update(network.layer(l).bias(port), firstMoment_.biasGradient(network.layer(l).name(), port),
						secondMoment_.biasGradient(network.layer(l).name(), port), network.layer(l).learningRateFactor());
			}
		}
	}

	for (u32 c = 0; c < network.nConnections(); c++) {
		if (network.connection(c).hasWeights() && network.connection(c).isTrainable()) {
			firstMoment_.weightsGradient(network.connection(c).name()).scale(beta1_);
			firstMoment_.weightsGradient(network.connection(c).name()).add(statistics.weightsGradient(network.connection(c).name()), (Float)(1.0 - beta1_));

			secondMoment_.weightsGradient(network.connection(c).name()).scale(beta2_);
			statistics.weightsGradient(network.connection(c).name()).elementwiseMultiplication(statistics.weightsGradient(network.connection(c).name()));
			secondMoment_.weightsGradient(network.connection(c).name()).add(statistics.weightsGradient(network.connection(c).name()), (Float)(1.0 - beta2_));

			update(network.connection(c).weights(), firstMoment_.weightsGradient(network.connection(c).name()),
					secondMoment_.weightsGradient(network.connection(c).name()), network.connection(c).learningRateFactor());
		}
	}
}
