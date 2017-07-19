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
 * Forwarder.hh
 *
 *  Created on: Jan 18, 2017
 *      Author: richard
 */

#ifndef NN_FORWARDER_HH_
#define NN_FORWARDER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "NeuralNetwork.hh"
#include "MinibatchGenerator.hh"
#include "TrainingCriteria.hh"
#include <Features/FeatureWriter.hh>

namespace Nn {

/*
 * only forwarding (unsupervised or supervised for classification error evaluation)
 * can be used for both, frame-wise forwarding and sequence forwarding
 */
class Forwarder
{
private:
	static const Core::ParameterEnum paramTask_;
	static const Core::ParameterEnum paramEvaluate_;
	enum Task { evaluate, dumpFeatures };
	enum Evaluation { none, classificationError, crossEntropy, squaredError };
protected:
	Task task_;
	Evaluation evaluation_;
	MinibatchGenerator minibatchGenerator_;
	NeuralNetwork network_;
	Features::FeatureWriter* writer_;
	Float evalResult_;
	u32 evalNormalization_;
	bool isInitialized_;

	// dump features
	void restoreSequenceOrder(const std::vector< Math::Matrix<Float> >& in, std::vector< Math::Matrix<Float> >& out);
	void dumpBatch(Matrix& source);
	void dumpBatch(MatrixContainer& source);
	void dumpSequenceBatch(MatrixContainer& source);
	// evaluation
	Float evaluateBatch(Matrix& source, Matrix& targets);
	Float evaluateSequenceBatch(MatrixContainer& source, Matrix& targets);
	Float evaluateSequenceBatch(MatrixContainer& source, MatrixContainer& targets);
	// tasks
	void dump();
	void eval();
public:
	Forwarder();
	virtual ~Forwarder();
	virtual void initialize();
	virtual void forward(u32 batchSize);
	virtual void finalize();
};

} // namespace

#endif /* NN_FORWARDER_HH_ */
