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
 * GrammarGenerator.cc
 *
 *  Created on: May 31, 2017
 *      Author: richard
 */

#include "GrammarGenerator.hh"

using namespace Hmm;


/*
 * GrammarGenerator
 */
const Core::ParameterEnum GrammarGenerator::paramGrammarType_("type", "finite, n-gram", "finite", "grammar");

const Core::ParameterString GrammarGenerator::paramGrammarFile_("file", "", "grammar");

GrammarGenerator::GrammarGenerator() :
		grammarFile_(Core::Configuration::config(paramGrammarFile_))
{}

void GrammarGenerator::write() {
	/* write result to file */
	if (grammarFile_.empty())
		Core::Error::msg("GrammarGenerator::generate(): grammar.file not specified.") << Core::Error::abort;
	if (!Core::Utils::isGz(grammarFile_))
		grammarFile_.append(".gz");
	Core::CompressedStream f(grammarFile_, std::ios::out);
	// write nonterminal symbols
	std::vector<std::string> tmp;
	for (std::set<std::string>::iterator it = nonterminals_.begin(); it != nonterminals_.end(); it++)
		tmp.push_back(*it);
	for (u32 i = 0; i < tmp.size() - 1; i++)
		f << tmp.at(i) << ", ";
	f << tmp.back() << Core::IOStream::endl;
	// write terminal symbols
	for (u32 i = 0; i < labelReader_.featureDimension()-1; i++)
		f << i << ", ";
	f << labelReader_.featureDimension()-1 << Core::IOStream::endl;
	// write rules
	for (u32 i = 0; i < rules_.size(); i++)
		f << rules_.at(i) << Core::IOStream::endl;
	f.close();
}

GrammarGenerator* GrammarGenerator::create() {
	switch ((GrammarType) Core::Configuration::config(paramGrammarType_)) {
	case finite:
		Core::Log::os("Create finite grammar generator.");
		return new FiniteGrammarGenerator();
		break;
	case ngram:
		Core::Log::os("Create n-gram grammar generator.");
		return new NGramGenerator();
		break;
	default:
		return 0; // this can not happen
	}
}


/*
 * FiniteGrammarGenerator
 */
void FiniteGrammarGenerator::depthFirstSearch(const Core::Tree<u32, Float>::Node& node, const std::string& prefix) {

	// store nonterminal and terminal symbol and generate rule
	std::stringstream n; n << prefix << "_" << node.key();
	std::stringstream t; t << node.key();
	nonterminals_.insert(n.str());
	std::stringstream rule;
	rule << prefix << " -> " << t.str() << " " << n.str() << " " << node.value();
	rules_.push_back(rule.str());

	// if this is a leaf, add rule for sequence end symbol
	if (node.nChildren() == 0) {
		std::stringstream endrule;
		endrule << n.str() << " -> . . 1";
		rules_.push_back(endrule.str());
	}
	// else proceed with the children
	else {
		for (u32 i = 0; i < node.nChildren(); i++) {
			depthFirstSearch(node.child(i), n.str());
		}
	}
}

void FiniteGrammarGenerator::generate() {

	/* create prefix tree with all label sequences */
	Core::Tree<u32, Float> tree(0, 1.0);
	labelReader_.initialize();
	while (labelReader_.hasSequences()) {
		std::vector<u32> path(labelReader_.nextLabelSequence());
		path.insert(path.begin(), 0);
		tree.addPath(path, 1.0);
	}

	/* traverse tree in depth-first order and create rules of the grammar */
	nonterminals_.insert("S");
	for (u32 i = 0; i < tree.root().nChildren(); i++)
		depthFirstSearch(tree.root().child(i), "S");

	write();
}


/*
 * NGramGenerator
 */
const Core::ParameterInt NGramGenerator::paramNGramOrder_("n-gram-order", 0, "grammar");

const Core::ParameterBool NGramGenerator::paramBackingOff_("backing-off", true, "grammar");

const NGramGenerator::Word NGramGenerator::root = -1;

const NGramGenerator::Word NGramGenerator::senStart = -2;

/*** LmTree ***/
NGramGenerator::LmTree::LmTree(const Key& rootKey, const Value& rootValue) :
		Precursor(rootKey, rootValue)
{}

void NGramGenerator::LmTree::unseenWords(const Context& history, u32 lexiconSize, std::vector<Word>& unseen) const {
	unseen.clear();
	// if history does not exist, N(h,w) = 0 for all w
	if (!pathExists(history)) {
		for (u32 w = 0; w < lexiconSize; w++)
			unseen.push_back((Word)w);
	}
	else {
		std::vector<bool> tmp(lexiconSize, false);
		const Node n = node(history);
		for (u32 i = 0; i < n.nChildren(); i++) {
			tmp.at(n.child(i).key()) = true;
		}
		for (u32 i = 0; i < lexiconSize; i++) {
			if (!tmp.at(i))
				unseen.push_back((Word)i);
		}
	}
}

NGramGenerator::Count NGramGenerator::LmTree::countSingletons(const Node& node, u32 level) const {
	if (level == 0) {
		if (node.value() == 1) return 1;
		else return 0;
	}
	else {
		Count sum = 0;
		for (u32 i = 0; i < node.nChildren(); i++) {
			sum += countSingletons(node.child(i), level-1);
		}
		return sum;
	}
}

NGramGenerator::Count NGramGenerator::LmTree::countSingletons(u32 level) const {
	return countSingletons(root_, level);
}
/*** End LmTree ***/


NGramGenerator::NGramGenerator() :
		nGramOrder_(Core::Configuration::config(paramNGramOrder_)),
		backingOff_(Core::Configuration::config(paramBackingOff_)),
		nWords_(0),
		lexiconSize_(0),
		lmTree_(root, 0),
		lambda_(nGramOrder_, 0.0)
{}

void NGramGenerator::accumulate(const std::vector<u32>& sequence) {
		for (u32 i = 0; i < sequence.size(); i++) {
			Context path(1, root); // each context/path starts with the root node
			for (s32 k = nGramOrder_; k >= 0; k--) {
				if ((s32)i < k)
					path.push_back(senStart);
				else
					path.push_back(sequence.at(i - k));
			}
			if (!lmTree_.pathExists(path))
				lmTree_.addPath(path, 0);
			lmTree_.value(path)++;
			while (path.size() > 1) {
				path.pop_back();
				lmTree_.value(path)++;
			}
		}
	nWords_ += sequence.size();
}

void NGramGenerator::estimateDiscountingParameter() {
	for (u32 historyLength = 0; historyLength < nGramOrder_; historyLength++) {
		lambda_.at(historyLength) = (Float)lmTree_.countSingletons(historyLength + 1) / (Float)nWords_;
	}
	Core::Log::openTag("linear-discounting");
	for (u32 historyLength = 0; historyLength < nGramOrder_; historyLength++) {
		Core::Log::os() << historyLength << "-gram discount: " << lambda_.at(historyLength);
	}
	Core::Log::closeTag();
}

Float NGramGenerator::probability(const Context& c) {
	require_gt(c.size(), 1);
	Context context = c;
	u32 historyLength = context.size() - 2;
	std::vector<Word> unseen;
	/* standard n-gram */
	if (lmTree_.pathExists(context)) {
		Float p = lmTree_.value(context);
		context.pop_back();
		p /= lmTree_.value(context);
		lmTree_.unseenWords(context, lexiconSize_, unseen);
		if (backingOff_ && (unseen.size() > 0)) // multiplication with backing-off parameter only if unseen events with the same history actually exist
			return (1 - lambda_.at(historyLength)) * p;
		else
			return p;
	}
	/* unseen event with backing-off */
	else if (backingOff_) {
		Word w = context.back();
		context.pop_back();
		std::vector<Word> unseen;
		lmTree_.unseenWords(context, lexiconSize_, unseen);
		context.push_back(w);
		context.erase(context.begin() + 1); // remove oldest history element
		// compute backing-off score p(w|h')
		Context tmpContext(context);
		Float p = probability(tmpContext);
		// compute renormalization for backing-off
		Float norm = 0;
		for (u32 i = 0; i < unseen.size(); i++) {
			context.back() = unseen.at(i);
			Context tmpContext(context);
			norm += probability(tmpContext);
		}
		// return renormalized probability, include backing-off parameter only if there are acutally seen events with the same history
		return (unseen.size() < lexiconSize_ ? lambda_.at(historyLength) : 1.0) * p / norm;
	}
	/* unseen event without backing-off */
	else {
		return 0.0;
	}
}

void NGramGenerator::extendContext(Context& c, std::vector<Context>& contexts) {
	if (c.size() < nGramOrder_ + 2) { // + 2 due to root and actual word
		// extend context by all words
		for (Word w = 0; w < (s32)lexiconSize_; w++) {
			c.push_back(w);
			extendContext(c, contexts);
			c.pop_back();
		}
		// also include senStart in the context history
		if ((c.size() < nGramOrder_ + 1) && (c.back() < 0)) { // c.back() < 0 is true for senStart and root
			c.push_back(senStart);
			extendContext(c, contexts);
			c.pop_back();
		}
	}
	else {
		contexts.push_back(c);
	}
}

void NGramGenerator::addRule(const Context& c) {
	std::stringstream s;
	s << "S";
	if (c.at(c.size() - 2) != senStart) {
		for (u32 i = 1; i < c.size()-1; i++)
			s << "_" << (c.at(i) < 0 ? lexiconSize_ : c.at(i));
	}
	nonterminals_.insert(s.str());
	s << " -> " << c.back() << " S";
	for (u32 i = 2; i < c.size(); i++)
		s << "_" << (c.at(i) < 0 ? lexiconSize_ : c.at(i));
	Float p = probability(c);
	s << " " << p;
	if (p > 0)
		rules_.push_back(s.str());
}

void NGramGenerator::generate() {
	labelReader_.initialize();
	lexiconSize_ = labelReader_.featureDimension();
	while (labelReader_.hasSequences()) {
		accumulate(labelReader_.nextLabelSequence());
	}
	if (backingOff_)
		estimateDiscountingParameter();

	// generate all possible contexts (w, h) and add the corresponding rule
	std::vector<Context> contexts;
	Context c(1, root);
	extendContext(c, contexts);
	for (u32 i = 0; i < contexts.size(); i++)
		addRule(contexts.at(i));

	// add end rules
	for (std::set<std::string>::iterator it = nonterminals_.begin(); it != nonterminals_.end(); ++it) {
		if (it->compare("S") != 0) {
			std::stringstream s;
			s << (*it) << " -> . . 1";
			rules_.push_back(s.str());
		}
	}

	write();
}
