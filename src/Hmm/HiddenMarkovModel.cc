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
 * HiddenMarkovModel.cc
 *
 *  Created on: Mar 4, 2015
 *      Author: richard
 */

#include "HiddenMarkovModel.hh"

using namespace Hmm;

/*
 * HiddenMarkovModel
 */
const Core::ParameterEnum HiddenMarkovModel::paramHmmType_("type", "standard-hmm, single-state-hmm", "standard-hmm", "hidden-markov-model");

const Core::ParameterString HiddenMarkovModel::paramHmmFile_("model-file", "", "hidden-markov-model");

const Core::ParameterString HiddenMarkovModel::paramTransitionProbabilityFile_("transition-probability-file", "", "hidden-markov-model");

HiddenMarkovModel::HiddenMarkovModel() :
		hmmFile_(Core::Configuration::config(paramHmmFile_)),
		transitionProbabilityFile_(Core::Configuration::config(paramTransitionProbabilityFile_)),
		nClasses_(0),
		nStates_(0),
		isInitialized_(false)
{}

void HiddenMarkovModel::initialize() {
	require(!isInitialized_);

	// load hmm definition
	if (hmmFile_.empty())
		Core::Error::msg("HiddenMarkovModel::initialize: hmm-file not specified.") << Core::Error::abort;
	statesPerClass_.read(hmmFile_);
	// determine start states
	nClasses_ = statesPerClass_.size();
	startStates_.resize(nClasses_);
	startStates_.at(0) = 0;
	for (u32 c = 1; c < nClasses_; c++)
		startStates_.at(c) = startStates_.at(c-1) + statesPerClass_.at(c-1);
	// compute mapping from state to class
	stateToClass_.resize(statesPerClass_.sum());
	u32 state = 0;
	for (u32 c = 0; c < statesPerClass_.size(); c++) {
		for (u32 s = 0; s < statesPerClass_.at(c); s++) {
			stateToClass_.at(state) = c;
			state++;
		}
	}
	nStates_ = stateToClass_.size();

	// load transition probabilities
	if (transitionProbabilityFile_.empty())
		Core::Error::msg("HiddenMarkovModel::initialize: transition-probability-file not specified.") << Core::Error::abort;
	loopScores_.read(transitionProbabilityFile_);
	forwardScores_.resize(loopScores_.size());
	forwardScores_.fill(1.0);
	forwardScores_.add(loopScores_, (Float)-1.0);
	loopScores_.log();
	forwardScores_.log();

	isInitialized_ = true;
}


u32 HiddenMarkovModel::nClasses() const {
	require(isInitialized_);
	return nClasses_;
}

u32 HiddenMarkovModel::nStates() const {
	require(isInitialized_);
	return nStates_;
}

u32 HiddenMarkovModel::nStates(u32 c) const {
	require(isInitialized_);
	require_lt(c, nClasses_);
	return statesPerClass_.at(c);
}

u32 HiddenMarkovModel::getClass(u32 state) const {
	require(isInitialized_);
	require_lt(state, nStates_);
	return stateToClass_.at(state);
}

u32 HiddenMarkovModel::startState(u32 c) const {
	require(isInitialized_);
	require_lt(c, nClasses_);
	return startStates_.at(c);
}

bool HiddenMarkovModel::isEndState(u32 state) const {
	require(isInitialized_);
	require_lt(state, nStates_);
	return ( (state == nStates_ - 1) || (stateToClass_.at(state) != stateToClass_.at(state+1)) );
}

u32 HiddenMarkovModel::successor(u32 state) const {
	require(isInitialized_);
	require_lt(state, nStates_);
	require(!isEndState(state));
	return state + 1;
}

Float HiddenMarkovModel::transitionScore(u32 stateFrom, u32 stateTo) const {
	require(isInitialized_);
	require_le(stateFrom, nStates_);
	if (stateFrom == stateTo) // loop transition
		return loopScores_.at(stateFrom);
	else if ((isEndState(stateFrom)) || (stateFrom + 1 == stateTo)) // forward transition
		return forwardScores_.at(stateFrom);
	else // invalid transition
		return -Types::inf<Float>();
}

HiddenMarkovModel* HiddenMarkovModel::create() {
	switch ((HmmType) Core::Configuration::config(paramHmmType_)) {
	case standardHmm:
		Core::Log::os("Create standard-hmm.");
		return new HiddenMarkovModel();
		break;
	case singleStateHmm:
		Core::Log::os("Create single-state-hmm.");
		return new SingleStateHiddenMarkovModel();
		break;
	default:
		return 0; // this can not happen
	}
}

/*
 * SingleStateHiddenMarkovModel
 */
const Core::ParameterInt SingleStateHiddenMarkovModel::paramNumberOfClasses_("number-of-classes", 0, "hidden-markov-model");

SingleStateHiddenMarkovModel::SingleStateHiddenMarkovModel() :
		Precursor()
{
	nClasses_ = Core::Configuration::config(paramNumberOfClasses_);
	nStates_ = nClasses_;
	require_gt(nClasses_, 0);
}

void SingleStateHiddenMarkovModel::initialize() {
	stateToClass_.resize(nStates_);
	for (u32 c = 0; c < nStates_; c++)
		stateToClass_.at(c) = c;
	statesPerClass_.resize(nClasses_);
	statesPerClass_.fill(1);
	startStates_.resize(nClasses_);
	for (u32 c = 0; c < nStates_; c++)
		startStates_.at(c) = c;
	loopScores_.resize(nStates_);
	loopScores_.fill(0.0);
	forwardScores_.resize(nStates_);
	forwardScores_.fill(0.0);
	isInitialized_ = true;
}
