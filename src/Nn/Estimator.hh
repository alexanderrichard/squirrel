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

#ifndef NN_ESTIMATOR_HH_
#define NN_ESTIMATOR_HH_

#include <Core/CommonHeaders.hh>
#include "NeuralNetwork.hh"
#include "Statistics.hh"
#include "LearningRateSchedule.hh"

namespace Nn {

/*
 * Base class for estimators
 * estimators are responsible for updating the model, e.g. based on the computed gradient
 */
class Estimator
{
private:
	static const Core::ParameterEnum paramEstimatorType_;
	static const Core::ParameterBool paramLogParameterNorm_;
	static const Core::ParameterBool paramLogGradientNorm_;
	static const Core::ParameterBool paramLogStepSize_;
	enum EstimatorType { none, steepestDescent, rprop };
protected:
	u32 epoch_;				// the current epoch number
	LearningRateSchedule* learningRateSchedule_;
	bool logParameterNorm_;
	bool logGradientNorm_;
	bool logStepSize_;
public:
	Estimator();
	virtual ~Estimator() {}
	virtual void initialize(NeuralNetwork& network);
	// tell the estimator to current epoch, useful to determine a learning rate
	virtual void setEpoch(u32 epoch);
	// each estimator will need a specific set of statistics, see Statistics.hh for possible choices
	virtual u32 requiredStatistics() const { return 0; }
	/*
	 * @param network the underlying neural network
	 * @param statistics the accumulated statistics for this model update, e.g. the gradient
	 */
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
	virtual void finalize() {}
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
private:
	Float biasWeight_;
public:
	SteepestDescentEstimator();
	virtual ~SteepestDescentEstimator() {}
	// stochasic gradient descent needs the gradient and some basics (e.g. number of observations)
	virtual u32 requiredStatistics() const { return (Statistics::baseStatistics | Statistics::gradientStatistics); }
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
	virtual void initialize(NeuralNetwork& network);
	// RProp needs the gradient and some basics (e.g. number of observations)
	virtual u32 requiredStatistics() const { return (Statistics::baseStatistics | Statistics::gradientStatistics); }
	virtual void estimate(NeuralNetwork& network, Statistics& statistics);
};


} // namespace

#endif /* NN_ESTIMATOR_HH_ */
