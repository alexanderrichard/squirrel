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
 * Trainer.hh
 *
 *  Created on: May 15, 2014
 *      Author: richard
 */

#ifndef NN_TRAINER_HH_
#define NN_TRAINER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "NeuralNetwork.hh"
#include "Estimator.hh"
#include "MinibatchGenerator.hh"

namespace Nn {

/*
 * Base class for neural network trainer
 */
class Trainer
{
private:
	static const Core::ParameterEnum paramTrainer_;
	static const Core::ParameterEnum paramTask_;
	static const Core::ParameterInt paramNumberOfEpochs_;
	static const Core::ParameterInt paramFirstEpoch_;
	static const Core::ParameterInt paramEpochLength_;
	static const Core::ParameterInt paramSaveFrequency_;
	enum TrainerType { none, feedForwardTrainer, rnnTrainer, bagOfWordsNetworkTrainer, specialRnnTrainer };
protected:
	TrainingTask task_;
	u32 numberOfEpochs_;
	u32 nProcessedEpochs_;
	u32 epoch_; // keep track in which epoch we are
	u32 epochLength_; // specify how many observations an epoch contains
	u32 saveFrequency_;

	u32 nProcessedMinibatches_;
	u32 nProcessedObservations_;

	NeuralNetwork network_;
	MinibatchGenerator minibatchGenerator_;
	Estimator* estimator_;

	bool isInitialized_;

public:
	Trainer();
	virtual ~Trainer();
	virtual void initialize();
	virtual void finalize();
	void processBatch(u32 batchSize);
	void processEpoch(u32 batchSize);
	void processAllEpochs(u32 batchSize);
	NeuralNetwork& network();
	Estimator& estimator();
	/* override this method for supervised frame-wise training */
	virtual void processBatch(Matrix& source, Matrix& target);
	/* override this method for supervised sequence training with one target per sequence */
	virtual void processSequenceBatch(MatrixContainer& source, Matrix& target);
	/* override this method for supervised sequence training where the targets are also sequences */
	virtual void processSequenceBatch(MatrixContainer& source, MatrixContainer& target);

	/* factory */
	static Trainer* createTrainer();
};

} // namespace

#endif /* NN_TRAINER_HH_ */
