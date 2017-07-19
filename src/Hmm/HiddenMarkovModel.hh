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
 * HiddenMarkovModel.hh
 *
 *  Created on: Mar 4, 2015
 *      Author: richard
 */

#ifndef HMM_HIDDENMARKOVMODEL_HH_
#define HMM_HIDDENMARKOVMODEL_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>

namespace Hmm {

/*
 * HiddenMarkovModel
 */
class HiddenMarkovModel
{
private:
	static const Core::ParameterEnum paramHmmType_;
	static const Core::ParameterString paramHmmFile_;
	static const Core::ParameterString paramTransitionProbabilityFile_;
	enum HmmType { standardHmm, singleStateHmm };
protected:
	std::string hmmFile_;
	std::string transitionProbabilityFile_;
	u32 nClasses_;
	u32 nStates_;
	Math::Vector<u32> stateToClass_;
	Math::Vector<u32> statesPerClass_;
	Math::Vector<u32> startStates_;
	Math::Vector<Float> loopScores_;
	Math::Vector<Float> forwardScores_;
	bool isInitialized_;
public:
	HiddenMarkovModel();
	virtual ~HiddenMarkovModel() {}
	/*
	 * load a hmm specified by the hmm-file and the transition-probability-file
	 */
	virtual void initialize();
	/*
	 * @return number of classes
	 */
	u32 nClasses() const;
	/*
	 * @return number of hmm states
	 */
	u32 nStates() const;
	/*
	 * @return number of hmm states for class c
	 */
	u32 nStates(u32 c) const;
	/*
	 * @return the class, state belongs to
	 */
	u32 getClass(u32 state) const;
	/*
	 * @return the start state of class c
	 */
	u32 startState(u32 c) const;
	/*
	 * @return true if the state is the last hmm state of its class
	 */
	bool isEndState(u32 state) const;
	/*
	 * @return the successor of state (unless state is an end-state, then abort with error)
	 */
	u32 successor(u32 state) const;
	/*
	 * @return log transition probability from stateFrom to stateTo
	 */
	Float transitionScore(u32 stateFrom, u32 stateTo) const;

	/*
	 * factory
	 */
	static HiddenMarkovModel* create();
};

/*
 * SingleStateHiddenMarkovModel
 * has only one state per class
 */
class SingleStateHiddenMarkovModel : public HiddenMarkovModel
{
private:
	typedef HiddenMarkovModel Precursor;
	static const Core::ParameterInt paramNumberOfClasses_;
public:
	SingleStateHiddenMarkovModel();
	virtual ~SingleStateHiddenMarkovModel() {}
	virtual void initialize();
};

} // namespace

#endif /* HMM_HIDDENMARKOVMODEL_HH_ */
