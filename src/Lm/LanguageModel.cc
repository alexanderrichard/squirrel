#include "LanguageModel.hh"
#include <cmath>

using namespace Lm;

/*
 * NGram
 */
const Word NGram::root = -1;

const Word NGram::senStart = -2;

const Core::ParameterInt NGram::paramLexiconSize_("lexicon-size", 0, "language-model");

const Core::ParameterBool NGram::paramBackingOff_("backing-off", true, "language-model");

const Core::ParameterString NGram::paramLanguageModelFile_("file", "", "language-model");

const Core::ParameterEnum NGram::paramLanguageModelType_("type", "zerogram, unigram, bigram, trigram", "unigram", "language-model");

/*** LmTree ***/
NGram::LmTree::LmTree(const Key& rootKey, const Value& rootValue) :
		Precursor(rootKey, rootValue)
{}

void NGram::LmTree::unseenWords(const Context& history, u32 lexiconSize, std::vector<Word>& unseen) const {
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

Count NGram::LmTree::countSingletons(const Node& node) const {
	if (node.isLeaf()) {
		if (node.value() == 1) return 1;
		else return 0;
	}
	else {
		Count sum = 0;
		for (u32 i = 0; i < node.nChildren(); i++) {
			sum += countSingletons(node.child(i));
		}
		return sum;
	}
}

Count NGram::LmTree::countSingletons() const {
	return countSingletons(root_);
}

void NGram::LmTree::saveSubtree(Core::BinaryStream& os, const Node& node) const {
	os << node.key();
	os << node.value();
	os << node.nChildren();
	for (u32 i = 0; i < node.nChildren(); i++) {
		saveSubtree(os, node.child(i));
	}
}

void NGram::LmTree::loadSubtree(Core::BinaryStream& os, Node& node) {
	Word key;
	Count value;
	u32 nChildren;
	os >> key;
	os >> value;
	os >> nChildren;
	node.key() = key;
	node.value() = value;
	for (u32 i = 0; i < nChildren; i++) {
		node.addChild(i, 0);
	}
	for (u32 i = 0; i < nChildren; i++) {
		loadSubtree(os, node.child(i));
	}
}

void NGram::LmTree::saveTree(Core::BinaryStream& os) const {
	saveSubtree(os, root_);
}

void NGram::LmTree::loadTree(Core::BinaryStream& os) {
	loadSubtree(os, root_);
}
/*** End LmTree ***/

NGram::NGram() :
		nGramOrder_(0),
		nWords_(0),
		lexiconSize_(Core::Configuration::config(paramLexiconSize_)),
		backingOff_(Core::Configuration::config(paramBackingOff_)),
		lmFile_(Core::Configuration::config(paramLanguageModelFile_)),
		lmType_((LmType) Core::Configuration::config(paramLanguageModelType_))
{
	require_gt(lexiconSize_, 0);
	switch (lmType_) {
	case zerogram:
		nGramOrder_ = 0;
		break;
	case unigram:
		nGramOrder_ = 1;
		break;
	case bigram:
		nGramOrder_ = 2;
		break;
	case trigram:
		nGramOrder_ = 3;
		break;
	default:
		; // this can not happen
	}
	lmTrees_.resize(nGramOrder_, LmTree(root, 0));
	lambda_.resize(nGramOrder_, 0.0);
}

void NGram::generateContexts(std::vector<Word>& sequence, std::vector<Context>& contexts, u32 historyLength) {
	require_gt(nGramOrder_, 0);
	contexts.clear();
	for (u32 i = 0; i < sequence.size(); i++) {
		require_lt(sequence.at(i), (s32)lexiconSize_);
		Context c(1, root); // each context/path starts with the root node
		for (u32 k = historyLength; k > 0; k--) {
			if (i < k)
				c.push_back(senStart);
			else
				c.push_back(sequence.at(i - k));
		}
		c.push_back(sequence.at(i));
		contexts.push_back(c);
	}
}

void NGram::accumulate(std::vector<Word>& sequence) {
	std::vector<Context> contexts;
	for (u32 historyLength = 0; historyLength < nGramOrder_; historyLength++) {
		generateContexts(sequence, contexts, historyLength);
		for (u32 i = 0; i < contexts.size(); i++) {
			lmTrees_.at(historyLength).addPath(contexts.at(i), 0);
			// increase count N(..., h2, h1, w)
			lmTrees_.at(historyLength).value(contexts.at(i))++;
			// increase count N(..., h2, h1)
			contexts.at(i).pop_back();
			lmTrees_.at(historyLength).value(contexts.at(i))++;
		}
	}
	nWords_ += sequence.size();
}

void NGram::estimateDiscountingParameter() {
	for (u32 historyLength = 0; historyLength < nGramOrder_; historyLength++) {
		lambda_.at(historyLength) = (Float)lmTrees_.at(historyLength).countSingletons() / (Float)nWords_;
	}
	Core::Log::openTag("language-model.linear-discounting");
	for (u32 historyLength = 0; historyLength < nGramOrder_; historyLength++) {
		Core::Log::os() << historyLength << "-gram discount: " << lambda_.at(historyLength);
	}
	Core::Log::closeTag();
}

Float NGram::probability(Context& context) const {
	require_gt(context.size(), 0);
	if (context.size() == 1) { // in case of a zero-gram
		return 1.0 / (Float)lexiconSize_;
	}
	else { // in case of higher orders
		u32 historyLength = context.size() - 2;
		std::vector<Word> unseen;
		/* standard n-gram */
		if (lmTrees_.at(historyLength).pathExists(context)) {
			Float p = lmTrees_.at(historyLength).value(context);
			context.pop_back();
			lmTrees_.at(historyLength).unseenWords(context, lexiconSize_, unseen);
			p /= lmTrees_.at(historyLength).value(context);
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
			lmTrees_.at(historyLength).unseenWords(context, lexiconSize_, unseen);
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
}

bool NGram::isValidWordIndex(Word w, bool includeSenStart) const {
	if ((w >= 0) && (w < (Word)lexiconSize_))
		return true;
	else if (includeSenStart && (w == senStart))
		return true;
	else
		return false;
}

void NGram::saveModel() {
	require(!lmFile_.empty());
	Core::Log::openTag("language-model");
	Core::Log::os("Save ") << nGramOrder_ << "-gram language model to " << lmFile_;
	Core::Log::closeTag();
	Core::BinaryStream os(lmFile_, std::ios::out);
	if (!os.is_open()) {
		std::cerr << "Error: NGram::saveModel: Could not open " << lmFile_ << std::endl;
		exit(1);
	}
	os << "#lm";
	os << nGramOrder_;
	for (u32 i = 0; i < nGramOrder_; i++) {
		os << lambda_.at(i);
		lmTrees_.at(i).saveTree(os);
	}
	os.close();
}

void NGram::loadModel() {
	require(!lmFile_.empty());
	Core::Log::openTag("language-model");
	Core::Log::os("Load ") << nGramOrder_ << "-gram language model from " << lmFile_;
	Core::Log::closeTag();

	Core::BinaryStream os(lmFile_, std::ios::in);
	if (!os.is_open()) {
		std::cerr << "Error: NGram::loadModel: Could not open " << lmFile_ << std::endl;
		exit(1);
	}
	char header[3];
	os >> header[0]; os >> header[1]; os >> header[2];
	if ((header[0] != '#') || (header[1] != 'l') || (header[2] != 'm')) {
		std::cerr << "Error: NGram::loadModel: header not correct. Abort." << std::endl;
		exit(1);
	}
	u32 nGramOrder;
	os >> nGramOrder;
	if (nGramOrder_ != nGramOrder) {
		std::cerr << "Error: NGram::loadModel: language model is a " << nGramOrder_ << "-gram, "
				"but model file " << lmFile_ << " is a " << nGramOrder << "-gram. Abort." << std::endl;
		exit(1);
	}
	for (u32 i = 0; i < nGramOrder_; i++) {
		os >> lambda_.at(i);
		lmTrees_.at(i).loadTree(os);
	}
	os.close();
}

Float NGram::probability() const {
	require_eq(nGramOrder_, 0);
	return 1.0 / (Float)lexiconSize_;
}

Float NGram::probability(Word w) const {
	require_eq(nGramOrder_, 1);
	require(isValidWordIndex(w, false));
	Context c(1, root);
	c.push_back(w);
	return probability(c);
}

Float NGram::probability(Word w, Word h) const {
	require_eq(nGramOrder_, 2);
	require(isValidWordIndex(w, false));
	require(isValidWordIndex(h));
	Context c(1, root);
	c.push_back(h);
	c.push_back(w);
	return probability(c);
}

Float NGram::probability(Word w, Word h, Word hh) const {
	require_eq(nGramOrder_, 3);
	require(isValidWordIndex(w, false));
	require(isValidWordIndex(h));
	require(isValidWordIndex(hh));
	Context c(1, root);
	c.push_back(hh);
	c.push_back(h);
	c.push_back(w);
	return probability(c);
}

Float NGram::logPerplexity(std::vector<Word>& sequence) {
	std::vector<Context> contexts;
	if (nGramOrder_ == 0) {
		return -std::log(1.0 / lexiconSize_);
	}
	else {
		generateContexts(sequence, contexts, nGramOrder_ - 1);
		Float logPP = 0;
		for (u32 i = 0; i < contexts.size(); i++) {
			logPP += std::log(probability(contexts.at(i)));
		}
		return -logPP / contexts.size();
	}
}
