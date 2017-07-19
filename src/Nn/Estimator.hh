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
 * Estimator.hh
 *
 *  Created on: May 21, 2014
 *      Author: richard
 */

#ifndef NN_ESTIMATOR_HH_
#define NN_ESTIMATOR_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "NeuralNetwork.hh"
#include "Statistics.hh"
#include "LearningRateSchedule.hh"

namespace Nn {

/*
 * Base class for estimators
 */
class Estimator
{
private:
	static const Core::ParameterEnum paramEstimatorType_;
	static const Core::ParameterBool paramLogParameterNorm_;
	static const Core::ParameterBool paramLogGradientNorm_;
	static const Core::ParameterBool paramLogStepSize_;
	enum EstimatorType { none, steepestDescent, rprop, adam };
protected:
	u32 epoch_;				// the current epoch number
	LearningRateSchedule* learningRateSchedule_;
	bool logParameterNorm_;
	bool logGradientNorm_;
	bool logStepSize_;
public:
	Estimator();
	virtual ~Estimator() {}
	virtual void initialize(NeuralNetwork& network, TrainingTask task);
	virtual void setEpoch(u32 epoch);
	virtual u32 requiredStatistics() const { return Statistics::none; }
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
	virtual void finalize();
	virtual Float learningRate() { return learningRateSchedule_->learningRate(); }
	/* factory */
	static Estimator* createEstimator();
};

/*
 * stochastic gradient descent estimator
 */
class SteepestDescentEstimator : public Estimator
{
private:
	typedef Estimator Precursor;
private:
	static const Core::ParameterFloat paramBiasWeight_;
	static const Core::ParameterBool paramUseMomentum_;
	static const Core::ParameterFloat paramMomentum_;
private:
	Float biasWeight_;
	bool useMomentum_;
	f32 momentum_;
	Statistics momentumStats_;
public:
	SteepestDescentEstimator();
	virtual ~SteepestDescentEstimator() {}
	virtual void initialize(NeuralNetwork& network, TrainingTask task);
	virtual u32 requiredStatistics() const { return Statistics::gradientStatistics; }
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
};

/*
 * rprop estimator
 */
class RpropEstimator : public Estimator
{
private:
	typedef Estimator Precursor;
private:
	static const Core::ParameterFloat paramIncreasingFactor_;
	static const Core::ParameterFloat paramDecreasingFactor_;
	static const Core::ParameterFloat paramMaxUpdateValue_;
	static const Core::ParameterFloat paramMinUpdateValue_;
	static const Core::ParameterFloat paramInitialStepSize_;
	static const Core::ParameterString paramLoadStepSizesFrom_;
	static const Core::ParameterString paramWriteStepSizesTo_;
private:
	Float increasingFactor_;
	Float decreasingFactor_;
	Float maxUpdateValue_;
	Float minUpdateValue_;
	Float initialStepSize_;
	Statistics oldGradients_;
	Statistics updateValues_;
public:
	RpropEstimator();
	virtual ~RpropEstimator();
	virtual void initialize(NeuralNetwork& network, TrainingTask task);
	virtual u32 requiredStatistics() const { return Statistics::gradientStatistics; }
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
};
/*
 * ADAM Estimator
 */
class AdamEstimator : public Estimator
{
private:
	typedef Estimator Precursor;
private:
	static const Core::ParameterFloat paramBeta1_;
	static const Core::ParameterFloat paramBeta2_;
	static const Core::ParameterFloat paramEpsilon_;

	Float beta1_;
	Float beta2_;
	Float epsilon_;
	u32 iteration_;

	Statistics firstMoment_;
	Statistics secondMoment_;
private:
	void update(Matrix &weights, const Matrix &firstMoment, const Matrix &secondMoment, Float learningRateFactor);
	void update(Vector &bias, const Vector &firstMoment, const Vector &secondMoment, Float learningRateFactor);

public:
	AdamEstimator();
	virtual ~AdamEstimator() { }
	virtual void initialize(NeuralNetwork& network, TrainingTask task);
	virtual u32 requiredStatistics() const { return Statistics::gradientStatistics; }
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
};

} // namespace

#endif /* NN_ESTIMATOR_HH_ */
