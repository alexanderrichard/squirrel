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
 * Grammar.cc
 *
 *  Created on: May 9, 2017
 *      Author: richard
 */

#include "Grammar.hh"
#include <map>

using namespace Hmm;

const Core::ParameterEnum Grammar::paramGrammarType_("type", "left-regular, single-path", "left-regular", "grammar");

const Core::ParameterString Grammar::paramGrammarFile_("file", "", "grammar");

Grammar::Grammar() :
		type_((GrammarType) Core::Configuration::config(paramGrammarType_)),
		grammarFile_(Core::Configuration::config(paramGrammarFile_)),
		nNonterminals_(0),
		nTerminals_(0),
		startSymbol_(0),
		endSymbol_(0),
		isInitialized_(false)
{}

void Grammar::initialize() {
	// open file
	if (grammarFile_.empty())
		Core::Error::msg("Grammar::initialize: grammar.file not specified.") << Core::Error::abort;
	Core::IOStream *f;
	if (Core::Utils::isGz(grammarFile_))
		f = new Core::CompressedStream(grammarFile_, std::ios::in);
	else
		f = new Core::AsciiStream(grammarFile_, std::ios::in);

	std::map<std::string, u32> nonterminals;
	std::map<std::string, u32> terminals;

	// read grammar
	std::string line;
	std::vector<std::string> tokenized;

	// first line: all non-terminals, first one is the start symbol
	f->getline(line);
	Core::Utils::tokenizeString(tokenized, line, ", ");
	for (u32 i = 0; i < tokenized.size(); i++)
		nonterminals[tokenized.at(i)] = i;
	nonterminals["."] = tokenized.size(); // add end symbol
	rules_.resize(nonterminals.size());
	endSymbol_ = tokenized.size();

	// second line: all terminals in correct order (first is mapped to label 0, second to label 1, ...)
	f->getline(line);
	Core::Utils::tokenizeString(tokenized, line, ", ");
	for (u32 i = 0; i < tokenized.size(); i++)
		terminals[tokenized.at(i)] = i;
	terminals["."] = tokenized.size(); // add end symbol

	// read rules
	while (f->getline(line)) {
		Core::Utils::tokenizeString(tokenized, line, " ->");
		if (tokenized.size() != 4)
			Core::Error::msg("Grammar::load: ") << line << " is an invalid rule definition." << Core::Error::abort;
		if (nonterminals.find(tokenized[0]) == nonterminals.end())
			Core::Error::msg("Grammar::load: ") << tokenized[0] << " not in list of nonterminal symbols." << Core::Error::abort;
		if (terminals.find(tokenized[1]) == terminals.end())
			Core::Error::msg("Grammar::load: ") << tokenized[1] << " not in list of terminal symbols." << Core::Error::abort;
		if (nonterminals.find(tokenized[2]) == nonterminals.end())
			Core::Error::msg("Grammar::load: ") << tokenized[2] << " not in list of nonterminal symbols." << Core::Error::abort;
		u32 label = terminals[tokenized[1]];
		u32 context = nonterminals[tokenized[2]];
		Float logProb = std::log(atof(tokenized[3].c_str()));
		rules_.at(nonterminals[tokenized[0]]).push_back(Rule(label, context, logProb));
	}

	delete f;

	nNonterminals_ = nonterminals.size();
	nTerminals_ = terminals.size();
	isInitialized_ = true;
}

const std::vector<Grammar::Rule>& Grammar::rules(u32 context) const {
	require(isInitialized_);
	require(context < nNonterminals_);
	return rules_.at(context);
}

Grammar* Grammar::create() {
	switch ((GrammarType) Core::Configuration::config(paramGrammarType_)) {
	case leftRegular:
		Core::Log::os("Create left-regular grammar.");
		return new Grammar();
		break;
	case singlePath:
		Core::Log::os("Create single-path grammar.");
		return new SinglePathGrammar();
		break;
	default:
		return 0; // this can not happen
	}
}


/*
 * SinglePathGrammar
 */
SinglePathGrammar::SinglePathGrammar() :
		Precursor()
{}

void SinglePathGrammar::setPath(const std::vector<u32>& path) {
	isInitialized_ = true;
	startSymbol_ = 0;
	endSymbol_ = path.size() + 1;
	nNonterminals_ = path.size() + 2; // + 2 due to start and end symbol

	// define rules such that only path can be derived
	u32 maxLabelIndex = 0;
	rules_.clear();
	rules_.resize(nNonterminals_);
	// define one rule per nonterminal symbol
	for (u32 i = 0; i < path.size(); i++) {
		maxLabelIndex = std::max(maxLabelIndex, path.at(i));
		rules_.at(i).push_back(Rule(path.at(i), i+1, 0.0));
	}
	nTerminals_ = maxLabelIndex + 1; // + 1 due to end symbol
	rules_.at(path.size()).push_back(Rule(nTerminals_, endSymbol_, 0.0)); // transition from last path element to end symbol
}
