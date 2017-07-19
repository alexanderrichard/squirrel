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
 * ViterbiDecoding.cc
 *
 *  Created on: May 10, 2017
 *      Author: richard
 */

#include "ViterbiDecoding.hh"

using namespace Hmm;

/*
 * TracebackNode
 */
ViterbiDecoding::TracebackNode::TracebackNode(u32 _state, TracebackNode* _predecessor) :
		predecessor(_predecessor),
		state(_state),
		nSuccessors(0),
		isActionBoundary(false)
{
	if (predecessor != 0)
		predecessor->nSuccessors++;
}

ViterbiDecoding::TracebackNode::~TracebackNode() {
	require_eq(nSuccessors, 0);
	if (predecessor != 0) {
		predecessor->nSuccessors--;
		if (predecessor->nSuccessors == 0)
			delete predecessor;
	}
}

/*
 * ViterbiDecoding::HypothesisKey
 */
bool ViterbiDecoding::HypothesisKey::disregardLength = false;

ViterbiDecoding::HypothesisKey::HypothesisKey(u32 _context, u32 _state, u32 _length) :
		context(_context),
		state(_state),
		length(_length)
{}

bool ViterbiDecoding::HypothesisKey::operator==(const HypothesisKey& k) const {
	return (state == k.state) && (context == k.context) && (disregardLength || (length == k.length));
}

u32 ViterbiDecoding::HypothesisKey::hash(const HypothesisKey& k) {
	if (disregardLength)
		return 31 * ((31 * k.context) ^ k.state);
	else
		return (31 * ((31 * k.context) ^ k.state)) ^ k.length;
}

/*
 * ViterbiDecoding::HypothesisNode
 */
ViterbiDecoding::HypothesisNode::HypothesisNode(u32 state) :
		score(-Types::inf<Float>()),
		traceback(new TracebackNode(state, 0))
{}

ViterbiDecoding::HypothesisNode::~HypothesisNode() {
	if ((traceback != 0) && (traceback->nSuccessors == 0))
		delete traceback;
}

void ViterbiDecoding::HypothesisNode::update(Float _score, TracebackNode* predecessorTraceback, bool isActionBoundary) {
	if (_score > score) {
		score = _score;
		traceback->isActionBoundary = isActionBoundary;
		if (traceback->predecessor != 0)
			traceback->predecessor->nSuccessors--;
		traceback->predecessor = predecessorTraceback;
		if (traceback->predecessor != 0)
			traceback->predecessor->nSuccessors++;
	}
}

/*
 * ViterbiDecoding::HypothesisList
 */
const Core::ParameterFloat ViterbiDecoding::HypothesisList::paramPruningThreshold_("pruning-threshold", Types::inf<Float>(), "viterbi-decoding");

const Core::ParameterInt ViterbiDecoding::HypothesisList::paramMaxHypotheses_("max-hypotheses", Types::max<u32>(), "viterbi-decoding");

ViterbiDecoding::HypothesisList::HypothesisList() :
		hashmap_(HypothesisKey::hash),
		pruningThreshold_(Core::Configuration::config(paramPruningThreshold_)),
		maxHypotheses_(Core::Configuration::config(paramMaxHypotheses_))
{
	require_ge(pruningThreshold_, 0);
}

bool ViterbiDecoding::HypothesisList::compare(Hypothesis* a, Hypothesis* b) {
	return a->value.score > b->value.score; // sort from best score to worst score
}

void ViterbiDecoding::HypothesisList::remove(const HypothesisKey& key) {
	hashmap_.remove(key);
}

void ViterbiDecoding::HypothesisList::update(const HypothesisKey& key, Float score, TracebackNode* predecessorTraceback, bool isActionBoundary) {
	Hypothesis *hypothesis = hashmap_.find(key);
	// insert if key does not yet exist and score is not too bad
	if (hypothesis == 0) {
		HypothesisNode value(key.state);
		// TODO remove this dirty hack for avoiding that the traceback pointer is deleted when the hash map calls the destructor of the copy of value during inserting
		TracebackNode* traceback = value.traceback;
		value.traceback = 0;
		hypothesis = hashmap_.insert(key, value);
		hypothesis->value.traceback = traceback;
	}
	if (hypothesis != 0) {
		if (score > hypothesis->value.score)
			hypothesis->key.length = key.length;
		hypothesis->value.update(score, predecessorTraceback, isActionBoundary);
	}
}

void ViterbiDecoding::HypothesisList::clear() {
	hashmap_.clear();
}

void ViterbiDecoding::HypothesisList::prune(u32 maxLength) {
	if (HypothesisKey::disregardLength)
		maxLength = 1;

	/* score threshold pruning */
	if (pruningThreshold_ < Types::inf<Float>()) {
		// store best scores
		std::vector< Float > bestScore(maxLength, -Types::inf<Float>());
		for (Hypothesis* h = hashmap_.begin(); h != hashmap_.end(); h = h->next) {
			u32 l = (HypothesisKey::disregardLength ? 0 : h->key.length-1);
			if (bestScore[l] < h->value.score) bestScore[l] = h->value.score;
		}
		// prune bad scores
		for (u32 l = 0; l < maxLength; l++) {
			// remove all bad hypotheses
			for (Hypothesis* h = hashmap_.begin(); h != hashmap_.end(); h = h->next) {
				if (h->value.score + pruningThreshold_ < bestScore[l])
					hashmap_.remove(h->key);
			}
		}
	}

	/* max hypotheses pruning */
	if (maxHypotheses_ < Types::max<u32>()) {
		// each length into one vector of hypotheses
		std::vector< std::vector<Hypothesis*> > hyp(maxLength);
		for (Hypothesis* h = hashmap_.begin(); h != hashmap_.end(); h = h->next) {
			u32 l = (HypothesisKey::disregardLength ? 0 : h->key.length-1);
			hyp.at(l).push_back(h);
		}
		// prune all but best maxHypotheses_ hypotheses
		for (u32 l = 0; l < maxLength; l++) {
			if (maxHypotheses_ < hyp[l].size()) {
				// sort hypotheses from best to worst score
				std::sort(hyp[l].begin(), hyp[l].end(), compare);
				// keep at most the best maxHypotheses nodes
				for (u32 i = maxHypotheses_; i < hyp[l].size(); i++) {
					hashmap_.remove(hyp[l].at(i)->key);
				}
			}
		}
	}
}

/*
 * ViterbiDecoding
 */
const Core::ParameterEnum ViterbiDecoding::paramViterbiOutput_("output", "hmm-states, labels", "labels", "viterbi-decoding");

const Core::ParameterFloat ViterbiDecoding::paramGrammarScale_("grammar-scale", 1.0, "viterbi-decoding");

const Core::ParameterFloat ViterbiDecoding::paramLengthModelScale_("length-model-scale", 1.0, "viterbi-decoding");

const Core::ParameterInt ViterbiDecoding::paramMaximalLength_("maximal-length", Types::max<u32>(), "viterbi-decoding");

ViterbiDecoding::ViterbiDecoding() :
		outputType_((ViterbiOutput) Core::Configuration::config(paramViterbiOutput_)),
		grammarScale_(Core::Configuration::config(paramGrammarScale_)),
		lengthModelScale_(Core::Configuration::config(paramLengthModelScale_)),
		maxLength_(Core::Configuration::config(paramMaximalLength_)),
		grammar_(0),
		scorer_(0),
		lengthModel_(0),
		hmm_(0),
		isInitialized_(false)
{}

ViterbiDecoding::~ViterbiDecoding() {
	delete grammar_;
	delete scorer_;
	delete lengthModel_;
	delete hmm_;
}

void ViterbiDecoding::initialize() {
	grammar_ = Grammar::create();
	grammar_->initialize();
	scorer_ = Scorer::create();
	scorer_->initialize();
	lengthModel_ = LengthModel::create();
	lengthModel_->initialize();
	if (lengthModel_->isFramewise())
		HypothesisKey::disregardLength = true;
	else
		HypothesisKey::disregardLength = false;
	hmm_ = HiddenMarkovModel::create();
	hmm_->initialize();
	isInitialized_ = true;
}

void ViterbiDecoding::sanityCheck() {
	// check number of hmm states
	if (hmm_->nStates() != scorer_->nClasses())
		Core::Error::msg("ViterbiDecoding::initialize: hmm state mismatch (") <<  hmm_->nStates() << " in hidden Markov model vs. "
		<< scorer_->nClasses() << " in scorer)." << Core::Error::abort;
	// grammar must have at most as many terminal symbols as there are classes (plus one sentence end symbol)
	if (hmm_->nClasses() + 1 < grammar_->nTerminals())
		Core::Error::msg("ViterbiDecoding::initialize: class mismatch (") << hmm_->nClasses() << " in hidden Markov model vs. "
		<< grammar_->nTerminals()-1 << " in grammar)." << Core::Error::abort;
}

void ViterbiDecoding::decodeFrame(u32 t, HypothesisList& oldHyp, HypothesisList& newHyp) {
	// for each hypothesis in oldHyp create the valid new hypotheses
	for (Hypothesis* h = oldHyp.begin(); h != oldHyp.end(); h = h->next) {
		/* loop transition (stay in the same hmm state) */
		HypothesisKey key(h->key.context, h->key.state, h->key.length + 1);
		Float score = h->value.score + hmm_->transitionScore(key.state, key.state)
				      + scorer_->frameScore(t, key.state) + lengthModelScale_ * lengthModel_->frameScore(key.length, key.state);
		if (HypothesisKey::disregardLength || (key.length <= maxLength_))
			newHyp.update(key, score, h->value.traceback);
		/* hmm end state: hypothesize action end and start new hypothesis using grammar */
		if (hmm_->isEndState(h->key.state)) {
			// ... start a new class hypothesis for each transition allowed by the grammar
			const std::vector<Grammar::Rule>& rules = grammar_->rules(h->key.context);
			for (u32 rule = 0; rule < rules.size(); rule++) {
				// exclude transitions to sequence end symbol as long as we are not at the end of the sequence
				if (rules.at(rule).context == grammar_->endSymbol())
					continue;
				HypothesisKey key(rules.at(rule).context, hmm_->startState(rules.at(rule).label), 1);
				score = h->value.score + lengthModelScale_ * lengthModel_->segmentScore(h->key.length, h->key.state)
						+ scorer_->segmentScore(t-1, h->key.length, h->key.state)
						+ grammarScale_ * rules.at(rule).logProbability + hmm_->transitionScore(h->key.state, key.state)
						+ scorer_->frameScore(t, key.state) + lengthModelScale_ * lengthModel_->frameScore(key.length, key.state);
				newHyp.update(key, score, h->value.traceback, true);
			}
		}
		/* no hmm end state: forward transition (loop already treated above) */
		else {
			// transition to next state
			HypothesisKey key(h->key.context, hmm_->successor(h->key.state), 1);
			score = h->value.score + lengthModelScale_ * lengthModel_->segmentScore(h->key.length, h->key.state)
					+ scorer_->segmentScore(t-1, h->key.length, h->key.state) + hmm_->transitionScore(h->key.state, key.state)
					+ scorer_->frameScore(t, key.state) + lengthModelScale_ * lengthModel_->frameScore(key.length, key.state);
			newHyp.update(key, score, h->value.traceback);
		}
	}
}

void ViterbiDecoding::traceback(HypothesisList& hyp, u32 sequenceLength) {
	// check if some valid path could be found
	segmentation_.clear();
	framewiseRecognition_.clear();
	if (hyp.nHypotheses() == 0) {
		Core::Log::os("No valid hypothesis found. Either sequence is too short or pruning is too strong.");
		segmentation_.push_back(ActionSegment(0, sequenceLength));
		framewiseRecognition_.resize(sequenceLength, 0);
	}
	// reconstruct decoded state- and label sequence
	else {
		// trace back best sequence
		u32 length = 0;
		for (TracebackNode* traceback = hyp.begin()->value.traceback->predecessor; traceback != 0; traceback = traceback->predecessor) {
			length++;
			if (outputType_ == hmmStates) {
				framewiseRecognition_.insert(framewiseRecognition_.begin(), traceback->state);
				if ((framewiseRecognition_.size() > 1) && (framewiseRecognition_[0] == framewiseRecognition_[1]))
					segmentation_[0].length++;
				else
					segmentation_.insert(segmentation_.begin(), ActionSegment(traceback->state, 1));
			}
			else { // outputType_ == labels
				framewiseRecognition_.insert(framewiseRecognition_.begin(), hmm_->getClass(traceback->state));
				if (traceback->isActionBoundary) {
					segmentation_.insert(segmentation_.begin(), ActionSegment(hmm_->getClass(traceback->state), length));
					length = 0;
				}
			}
		}
	}
}

Float ViterbiDecoding::decode(const Math::Matrix<Float>& sequence) {
	require(isInitialized_);
	u32 T = sequence.nColumns();
	scorer_->setSequence(sequence);

	HypothesisList oldHyp;
	HypothesisList newHyp;

	// create initial hypotheses
	const std::vector<Grammar::Rule>& rules = grammar_->rules(grammar_->startSymbol());
	for (u32 rule = 0; rule < rules.size(); rule++) {
		HypothesisKey key(rules.at(rule).context, hmm_->startState(rules.at(rule).label), 1);
		Float score = grammarScale_ * rules.at(rule).logProbability + scorer_->frameScore(0, key.state) + lengthModelScale_ * lengthModel_->frameScore(key.length, key.state);
		oldHyp.update(key, score, 0, true);
	}

	// decode all remaining frames
	for (u32 t = 1; t < T; t++) {
		// viterbi decoding of frame t
		decodeFrame(t, oldHyp, newHyp);
		// oldHyp must be cleared before nodes in newHyp are deleted due to traceback pointers!
		oldHyp.clear();
		// prune
		newHyp.prune(maxLength_);
		// swap old and new hypotheses for processing of next frame
		std::swap(oldHyp, newHyp);
	}

	// find best hypothesis (among all hypotheses that allow a transition to the sequence end symbol)
	for (Hypothesis* h = oldHyp.begin(); h != oldHyp.end(); h = h->next) {
		if (hmm_->isEndState(h->key.state)) {
			const std::vector<Grammar::Rule>& rules = grammar_->rules(h->key.context);
			for (u32 rule = 0; rule < rules.size(); rule++) {
				if (rules.at(rule).context == grammar_->endSymbol()) {
					// hmm transition probability: h->key.state+1 may not exist but does not matter since h->key.state is an end-state
					HypothesisKey key(grammar_->endSymbol(), Types::max<u32>(), 1);
					Float score = h->value.score + grammarScale_ * rules.at(rule).logProbability + hmm_->transitionScore(h->key.state, h->key.state+1)
								  + lengthModelScale_ * lengthModel_->segmentScore(h->key.length, h->key.state)
								  + scorer_->segmentScore(T-1, h->key.length, h->key.state);
					newHyp.update(key, score, h->value.traceback);
				}
			}
		}
	}

	Float score = (newHyp.nHypotheses() == 0) ? -Types::inf<Float>() : newHyp.begin()->value.score;
	traceback(newHyp, T);

	oldHyp.clear();
	newHyp.clear();

	return score;
}

Float ViterbiDecoding::realign(const Math::Matrix<Float>& sequence, const std::vector<u32>& labelSequence) {
	require(isInitialized_);
	if (grammar_->type() != Grammar::singlePath)
		Core::Error::msg("ViterbiDecoding::realign: grammar needs to be of type single-path for realignment.") << Core::Error::abort;
	dynamic_cast<SinglePathGrammar*>(grammar_)->setPath(labelSequence);
	sanityCheck();
	return decode(sequence);
}

u32 ViterbiDecoding::nOutputClasses() const {
	if (outputType_ == hmmStates)
		return hmm_->nStates();
	else // outputType_ == labels
		return hmm_->nClasses();
}
