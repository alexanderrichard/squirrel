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
 * Scorer.hh
 *
 *  Created on: May 12, 2017
 *      Author: richard
 */

#ifndef HMM_SCORER_HH_
#define HMM_SCORER_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Matrix.hh>
#include <Math/Vector.hh>
#include <Nn/NeuralNetwork.hh>

namespace Hmm {

class Scorer
{
private:
	static const Core::ParameterEnum paramScorerType_;
	enum ScorerType { framewiseNeuralNetworkScorer, segmentScorer };
protected:
	u32 nClasses_;
	bool isInitialized_;
	Math::Matrix<Float> const * sequence_;
public:
	Scorer();
	virtual ~Scorer() {}
	virtual void initialize();
	u32 nClasses() const { require(isInitialized_); return nClasses_; }
	virtual void setSequence(const Math::Matrix<Float>& sequence);
	virtual Float frameScore(u32 t, u32 c) { return 0.0; }
	virtual Float segmentScore(u32 t, u32 length, u32 c) { return 0.0; }

	/*
	 * factory
	 */
	static Scorer* create();
};


class FramewiseNeuralNetworkScorer : public Scorer
{
private:
	typedef Scorer Precursor;
	static const Core::ParameterString paramPriorFile_;
	static const Core::ParameterFloat paramPriorScale_;
	static const Core::ParameterBool paramLogarithmizeNetworkOutput_;
	static const Core::ParameterInt paramBatchSize_;
protected:
	std::string priorFile_;
	Float priorScale_;
	bool logarithmizeNetworkOutput_;
	u32 batchSize_;
	Nn::Vector prior_;
	Nn::NeuralNetwork network_;
	Nn::Matrix scores_;
public:
	FramewiseNeuralNetworkScorer();
	virtual ~FramewiseNeuralNetworkScorer() {}
	virtual void initialize();
	virtual void setSequence(const Math::Matrix<Float>& sequence);
	virtual Float frameScore(u32 t, u32 c);
};


class SegmentScorer : public FramewiseNeuralNetworkScorer
{
private:
	static const Core::ParameterBool paramScaleSegmentByLength_;
	typedef FramewiseNeuralNetworkScorer Precursor;
	bool scaleSegmentByLength_;
	Math::Matrix<Float> sequence_;
	Nn::Matrix networkInput_;
	u32 t_start_; // start frame for which scores have been precomputed
	u32 t_end_; // end frame for which scores have been precomputed
	void generateVector(u32 t_start, u32 t_end, u32 column, Nn::Matrix& result);
public:
	SegmentScorer();
	virtual ~SegmentScorer() {}
	virtual void initialize();
	virtual void setSequence(const Math::Matrix<Float>& sequence);
	virtual Float frameScore(u32 t, u32 c) { return 0.0; }
	virtual Float segmentScore(u32 t, u32 length, u32 c);
};

} // namespace

#endif /* HMM_SCORER_HH_ */
