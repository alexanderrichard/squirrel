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

#ifndef NN_TRAINER_HH_
#define NN_TRAINER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "NeuralNetwork.hh"
#include "Estimator.hh"
#include "Statistics.hh"
#include "TrainingCriteria.hh"
#include <Features/FeatureWriter.hh>

namespace Nn {

/*
 * Base class for neural network trainer
 */
class Trainer
{
private:
	static const Core::ParameterEnum paramTrainer_;
	enum TrainerType { dummy, forwarder, feedForwardTrainer, rnnTrainer };
protected:
	NeuralNetwork network_;
	Estimator* estimator_;
	u32 epoch_; // keep track in which epoch we are
	u32 epochLength_; // specify how many observations an epoch contains
	bool isInitialized_;

public:
	Trainer();
	virtual ~Trainer();
	/*
	 * initialize the trainer
	 * @param epochLength the number of observations until the epoch ends
	 */
	virtual void initialize(u32 epochLength = 0);
	virtual void finalize();
	// return whether or not the trainer can be used for labeled/unlabeled data
	virtual bool isSupervised() const = 0;
	virtual bool isUnsupervised() const = 0;
	/*
	 * @return the neural network this trainer operates on
	 */
	NeuralNetwork& network();
	/*
	 * @return the estimator used for model updates
	 */
	Estimator& estimator();
	/* set the epoch we are in */
	void setEpoch(u32 epoch);
	/* override this method for unsupervised frame-wise training */
	virtual void processBatch(Matrix& batch) {}
	/* override this method for unsupervised sequence training */
	virtual void processSequenceBatch(MatrixContainer& batchedSequence) {}
	/* override this method for supervised frame-wise training */
	virtual void processBatch(Matrix& batch, LabelVector& labels) {}
	/* override this method for supervised sequence training */
	virtual void processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels) {}

	/* factory */
	static Trainer* createFramewiseTrainer();
	static Trainer* createSequenceTrainer();
};

/*
 * dummy trainer, does nothing
 */
class DummyTrainer : public Trainer
{
public:
	DummyTrainer() : Trainer() {}
	virtual ~DummyTrainer() {}
	virtual bool isSupervised() const { return true; }
	virtual bool isUnsupervised() const { return true; }
};

/*
 * only forwarding (unsupervised or supervised for classification error evaluation)
 * can be used for both, frame-wise forwarding and sequence forwarding
 */
class Forwarder : public Trainer
{
private:
	static const Core::ParameterBool paramCacheNeuralNetworkOutput_;
	static const Core::ParameterBool paramLogConfusionMatrix_;
	typedef Trainer Precursor;
protected:
	Statistics statistics_;
	bool cacheNeuralNetworkOutput_;
	bool logConfusionMatrix_;
	Features::FeatureWriter featureWriter_;
	Math::Matrix<u32> confusionMatrix_;

	void updateStatistics(LabelVector& labels);
	void cacheNetworkOutput();
public:
	Forwarder();
	virtual ~Forwarder();
	virtual void initialize(u32 epochLength = 0);
	virtual void finalize();
	virtual bool isSupervised() const { return true; }
	virtual bool isUnsupervised() const { return true; }
	virtual void processBatch(Matrix& batch);
	virtual void processBatch(Matrix& batch, LabelVector& labels);
	virtual void processSequenceBatch(MatrixContainer& batchedSequence);
	virtual void processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels);
};

} // namespace

#endif /* NN_TRAINER_HH_ */
