#ifndef LM_LANGUAGEMODEL_HH_
#define LM_LANGUAGEMODEL_HH_

#include <Core/CommonHeaders.hh>
#include <Core/Tree.hh>

namespace Lm {

typedef s32 Word;
typedef u32 Count;

class NGram
{
protected:
	typedef Core::Tree<Word, Count>::Path Context;
	/*** LmTree ***/
	class LmTree : public Core::Tree<Word, Count>
	{
	private:
		typedef Core::Tree<Word, Count> Precursor;
		Count countSingletons(const Node& node) const;
	public:
		LmTree(const Key& rootKey, const Value& rootValue);
		void unseenWords(const Context& history, u32 lexiconSize_, std::vector<Word>& unseen) const;
		Count countSingletons() const;
		void saveSubtree(Core::BinaryStream& os, const Node& node) const;
		void loadSubtree(Core::BinaryStream& os, Node& node);
		void saveTree(Core::BinaryStream& os) const;
		void loadTree(Core::BinaryStream& os);
	};
	/*** End LmTree ***/

	static const Word root;
private:
	static const Core::ParameterInt paramLexiconSize_;
	static const Core::ParameterBool paramBackingOff_;
	static const Core::ParameterString paramLanguageModelFile_;
	static const Core::ParameterEnum paramLanguageModelType_;
	enum LmType { zerogram, unigram, bigram, trigram };
public:
	static const Word senStart;
protected:
	// one tree for each n-gram order
	// allows to read counts N(w, h1, h2, ...) via a context/path (root, ..., h2, h1, w)
	u32 nGramOrder_;
	u32 nWords_; // number of distinct words in the corpus
	u32 lexiconSize_; // number of words in the dictionary
	bool backingOff_;
	std::string lmFile_;
	LmType lmType_;
	std::vector<LmTree> lmTrees_;
	std::vector<Float> lambda_;
	void generateContexts(std::vector<Word>& sequence, std::vector<Context>& contexts, u32 historyLength);
	Float probability(Context& context) const;
	bool isValidWordIndex(Word w, bool includeSenStart = true) const;
public:
	NGram();
	virtual ~NGram() {}
	/*
	 * accumulate the statistics needed to compute the n-gram probabilities
	 * @sequence the n-gram statistics of this class sequence are added to the existing statistics
	 */
	void accumulate(std::vector<Word>& sequence);
	/*
	 * estimate the discounting parameter to account for unseen events
	 */
	void estimateDiscountingParameter();
	u32 lexiconSize() const { return lexiconSize_; }
	/*
	 * load/save the model specified by paramLanguageModelFile_
	 */
	void saveModel();
	void loadModel();
	u32 nGramOrder() const { return nGramOrder_; }
	/*
	 * probability functions
	 * @return probability p(w|h, hh, ...), i.e. probability of occurrence of (.., hh, h, w)
	 */
	// zerogram
	Float probability() const;
	// unigram
	Float probability(Word w) const;
	// bigram
	Float probability(Word w, Word h) const;
	// trigram
	Float probability(Word w, Word h, Word hh) const;

	/*
	 * @return the log perplexity of the given sequence, normalized over the number of words in the sequence
	 */
	Float logPerplexity(std::vector<Word>& sequence);
};

} // namespace

#endif /* LANGUAGEMODEL_HH_ */
