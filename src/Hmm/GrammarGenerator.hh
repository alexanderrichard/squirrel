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
 * GrammarGenerator.hh
 *
 *  Created on: May 31, 2017
 *      Author: richard
 */

#ifndef HMM_GRAMMARGENERATOR_HH_
#define HMM_GRAMMARGENERATOR_HH_

#include <Core/CommonHeaders.hh>
#include <Core/Tree.hh>
#include <set>
#include <sstream>
#include <Features/FeatureReader.hh>

namespace Hmm {

/*
 * GrammarGenerator
 * generates grammars from sequence label files
 */
class GrammarGenerator
{
private:
	static const Core::ParameterEnum paramGrammarType_;
	static const Core::ParameterString paramGrammarFile_;
	enum GrammarType { finite, ngram };
protected:
	std::string grammarFile_;
	Features::SequenceLabelReader labelReader_;
	std::vector< std::string > rules_;
	std::set< std::string > nonterminals_;
	void write();
public:
	GrammarGenerator();
	virtual ~GrammarGenerator() {}
	virtual void generate() = 0;

	static GrammarGenerator* create();
};


/*
 * FiniteGrammarGenerator
 */
class FiniteGrammarGenerator : public GrammarGenerator
{
private:
	typedef GrammarGenerator Precursor;
	void depthFirstSearch(const Core::Tree<u32, Float>::Node& node, const std::string& prefix);
public:
	FiniteGrammarGenerator() {}
	virtual ~FiniteGrammarGenerator() {}
	virtual void generate();
};


/*
 * FiniteGrammarGenerator
 */
class NGramGenerator : public GrammarGenerator
{
private:
	typedef s32 Word;
	typedef u32 Count;
	typedef Core::Tree<Word, Count>::Path Context;
	/*** LmTree ***/
	class LmTree : public Core::Tree<Word, Count>
	{
	private:
		typedef Core::Tree<Word, Count> Precursor;
		Count countSingletons(const Node& node, u32 level) const;
	public:
		LmTree(const Key& rootKey, const Value& rootValue);
		void unseenWords(const Context& history, u32 lexiconSize_, std::vector<Word>& unseen) const;
		Count countSingletons(u32 level) const;
	};
	/*** End LmTree ***/

	static const Word root;
private:
	typedef GrammarGenerator Precursor;
	static const Core::ParameterInt paramNGramOrder_;
	static const Core::ParameterBool paramBackingOff_;
	static const Word senStart;
protected:
	// one tree for each n-gram order
	// allows to read counts N(w, h1, h2, ...) via a context/path (root, ..., h2, h1, w)
	u32 nGramOrder_;
	bool backingOff_;
	u32 nWords_; // number of distinct words in the corpus
	u32 lexiconSize_; // number of words in the dictionary
	LmTree lmTree_;
	std::vector<Float> lambda_;
	Float probability(const Context& c);
	void accumulate(const std::vector<u32>& sequence);
	void estimateDiscountingParameter();
	void extendContext(Context& c, std::vector<Context>& contexts);
	void addRule(const Context& c);
public:
	NGramGenerator();
	virtual ~NGramGenerator() {}
	virtual void generate();
};

} // namespace

#endif /* HMM_GRAMMARGENERATOR_HH_ */
