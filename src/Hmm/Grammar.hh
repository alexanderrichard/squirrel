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
 * Grammar.hh
 *
 *  Created on: May 9, 2017
 *      Author: richard
 */

#ifndef HMM_GRAMMAR_HH_
#define HMM_GRAMMAR_HH_

#include <Core/CommonHeaders.hh>
#include <Core/Tree.hh>
#include <set>
#include <sstream>
#include <Features/FeatureReader.hh>

namespace Hmm {
 /*
  * class for left-regular grammars
  * file format is as follows (terminal/nonterminal symbols must not include "-", ">", ".", or ","):
  * N0, N1, ...      (list of nonterminal symbols, first one is start symbol)
  * T0, T1, T2, ...  (list of terminal symbols, end symbol is always "." and should not be defined in this line)
  * N0 -> 0 N1 0.5   (rule of the form "Nonterminal -> Terminal Nonterminal Probability")
  * ...              (more rules)
  * N0 -> . . 1.0    (example rule for the end symbol)
  */
class Grammar
{
public:
	struct Rule {
		u32 label;
		u32 context;
		Float logProbability;
		Rule(u32 _label, u32 _context, Float _probability) : label(_label), context(_context), logProbability(_probability) {}
	};
	enum GrammarType { leftRegular, singlePath };
private:
	static const Core::ParameterEnum paramGrammarType_;
	static const Core::ParameterString paramGrammarFile_;

protected:
	GrammarType type_;
	std::string grammarFile_;
	u32 nNonterminals_;
	u32 nTerminals_;
	u32 startSymbol_;
	u32 endSymbol_;
	std::vector< std::vector<Rule> > rules_;
	bool isInitialized_;

public:
	Grammar();
	virtual ~Grammar() {}
	virtual void initialize();
	GrammarType type() const { return type_; }
	u32 nNonterminals() const { require(isInitialized_); return nNonterminals_; }
	u32 nTerminals() const { require(isInitialized_); return nTerminals_; }
	u32 startSymbol() const { require(isInitialized_); return startSymbol_; } // start symbol is always mapped to 0
	u32 endSymbol() const { require(isInitialized_); return endSymbol_; } // sequence end symbol is a terminal symbol
	const std::vector<Rule>& rules(u32 context) const;

	/*
	 * factory
	 */
	static Grammar* create();
};


/*
 * grammar contains a single label path only
 */
class SinglePathGrammar : public Grammar
{
private:
	typedef Grammar Precursor;
public:
	SinglePathGrammar();
	virtual void initialize() {};
	virtual void setPath(const std::vector<u32>& path);
};

} // namespace

#endif /* HMM_GRAMMAR_HH_ */
