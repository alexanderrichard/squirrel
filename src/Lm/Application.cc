#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include "Application.hh"
#include "LanguageModel.hh"
#include "SentenceGenerator.hh"

using namespace Lm;

APPLICATION(Lm::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, build, perplexity, generate-sentences",
		"none");

const Core::ParameterString Application::paramCorpus_("corpus", "");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case build:
		trainLanguageModel();
		break;
	case perplexity:
		computePerplexity();
		break;
	case generateSentences:
		{
		SentenceGenerator s;
		s.initialize();
		s.logSentences();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}

/*
 * trains a language model
 * corpus format:
 * an ascii file containing in each line a sequence of word indices
 * word indices must range from 0 to lexicon-size - 1
 */
void Application::trainLanguageModel() {
	std::string corpusFile = Core::Configuration::config(paramCorpus_);
	require(!corpusFile.empty());

	NGram lm;

	// load training data
	std::ifstream corpus(corpusFile.c_str());
	require(corpus.is_open());
	std::string line;
	while (std::getline(corpus, line)) {
		std::vector<std::string> tmp;
		std::vector<Lm::Word> sequence;
		Core::Utils::tokenizeString(tmp, line);
		for (u32 i = 0; i < tmp.size(); i++) {
			Lm::Word w = atoi(tmp.at(i).c_str());
			if ((w < 0) || (w > (Lm::Word)lm.lexiconSize())) {
				std::cerr << "Error: Application::trainLanguageModel: word indices must only be in the range [0,...,lexicon-size-1]. Abort." << std::endl;
				exit(1);
			}
			else {
				sequence.push_back(w);
			}
		}
		// accumulate sequence
		lm.accumulate(sequence);
	}

	lm.estimateDiscountingParameter();
	lm.saveModel();
}

/*
 * computes the perplexity of a language model
 * corpus format:
 * an ascii file containing in each line a sequence of word indices
 * word indices must range from 0 to lexicon-size - 1
 */
void Application::computePerplexity() {
	std::string corpusFile = Core::Configuration::config(paramCorpus_);
	require(!corpusFile.empty());

	NGram lm;
	lm.loadModel();
	Float logPP = 0;
	u32 nWords = 0;

	// load corpus
	std::ifstream corpus(corpusFile.c_str());
	require(corpus.is_open());
	std::string line;
	while (std::getline(corpus, line)) {
		std::vector<std::string> tmp;
		std::vector<Lm::Word> sequence;
		Core::Utils::tokenizeString(tmp, line);
		for (u32 i = 0; i < tmp.size(); i++) {
			Lm::Word w = atoi(tmp.at(i).c_str());
			if ((w < 0) || (w > (Lm::Word)lm.lexiconSize())) {
				std::cerr << "Error: Application::computePerplexity: word indices must only be in the range [0,...,lexicon-size-1]. Abort." << std::endl;
				exit(1);
			}
			else {
				sequence.push_back(w);
			}
		}
		// compute log perplexity of the sequence
		logPP += lm.logPerplexity(sequence) * sequence.size();
		nWords += sequence.size();
	}

	Core::Log::os("Perplexity is ") << std::exp(logPP / nWords) << ".";
}
