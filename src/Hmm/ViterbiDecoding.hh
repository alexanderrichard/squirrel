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
 * ViterbiDecoding.hh
 *
 *  Created on: May 10, 2017
 *      Author: richard
 */

#ifndef HMM_VITERBIDECODING_HH_
#define HMM_VITERBIDECODING_HH_

#include <Core/CommonHeaders.hh>
#include <Core/HashMap.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>
#include <algorithm>
#include "Grammar.hh"
#include "Scorer.hh"
#include "LengthModel.hh"
#include "HiddenMarkovModel.hh"

namespace Hmm {

class ViterbiDecoding
{
public:
	struct ActionSegment {
		u32 label;
		u32 length;
		ActionSegment(u32 _label, u32 _length) : label(_label), length(_length) {}
	};

private:
	/* construct linked list as bookkeeping for hypotheses */

	/* traceback node */
	class TracebackNode {
	public:
		TracebackNode* predecessor;
		u32 state;
		u32 nSuccessors;
		bool isActionBoundary;
		TracebackNode(u32 _state, TracebackNode* _predecessor);
		~TracebackNode();
	};

	/* a key for the hash map */
	class HypothesisKey {
	public:
		static bool disregardLength;
		u32 context;
		u32 state;
		u32 length;
		HypothesisKey(u32 _context, u32 _state, u32 _length);
		bool operator==(const HypothesisKey& k) const;
		static u32 hash(const HypothesisKey& k);
	};

	/* a node in the list */
	class HypothesisNode {
	public:
		Float score;
		TracebackNode* traceback;
		HypothesisNode(u32 state);
		~HypothesisNode();
		void update(Float score, TracebackNode* predecessorTraceback, bool isActionBoundary = false);
	};

	/* the list */
	class HypothesisList {
	private:
		static const Core::ParameterFloat paramPruningThreshold_;
		static const Core::ParameterInt paramMaxHypotheses_;
	public:
		typedef Core::HashMap< HypothesisKey, HypothesisNode >::Entry Hypothesis;
	private:
		Core::HashMap< HypothesisKey, HypothesisNode > hashmap_;
		Float pruningThreshold_;
		u32 maxHypotheses_;
		static bool compare(Hypothesis* a, Hypothesis* b);
		void remove(const HypothesisKey& key);
	public:
		HypothesisList();
		u32 nHypotheses() const { return hashmap_.size(); }
		void update(const HypothesisKey& key, Float score, TracebackNode* predecessorTraceback, bool isActionBoundary = false);
		void clear();
		Hypothesis* begin() { return hashmap_.begin(); }
		Hypothesis* end() { return hashmap_.end(); }
		void prune(u32 maxLength);
	};

	/* the class for the actual Viterbi decoding */
private:
	static const Core::ParameterEnum paramViterbiOutput_;
	static const Core::ParameterFloat paramGrammarScale_;
	static const Core::ParameterFloat paramLengthModelScale_;
	static const Core::ParameterInt paramMaximalLength_;
	enum ViterbiOutput { hmmStates, labels };
	typedef HypothesisList::Hypothesis Hypothesis;
protected:
	ViterbiOutput outputType_;
	Float grammarScale_;
	Float lengthModelScale_;
	u32 maxLength_;
	Grammar* grammar_;
	Scorer* scorer_;
	LengthModel* lengthModel_;
	HiddenMarkovModel* hmm_;
	std::vector<ActionSegment> segmentation_;
	std::vector<u32> framewiseRecognition_;
	bool isInitialized_;
private:
	void decodeFrame(u32 t, HypothesisList& oldHyp, HypothesisList& newHyp);
	void traceback(HypothesisList& hyp, u32 sequenceLength);
public:
	ViterbiDecoding();
	virtual ~ViterbiDecoding();
	void initialize();
	void sanityCheck();
	Float decode(const Math::Matrix<Float>& sequence);
	Float realign(const Math::Matrix<Float>& sequence, const std::vector<u32>& labelSequence);
	const std::vector<ActionSegment>& segmentation() const { return segmentation_; }
	const std::vector<u32>& framewiseRecognition() const { return framewiseRecognition_; }
	u32 nOutputClasses() const;
};

} // namespace

#endif /* HMM_VITERBIDECODING_HH_ */
