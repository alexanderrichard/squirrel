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

#include "SentenceGenerator.hh"
#include <Math/Random.hh>
#include <sstream>

using namespace Lm;

const Core::ParameterInt SentenceGenerator::paramNSentences_("number-of-sentences", 1, "sentence-generator");

const Core::ParameterInt SentenceGenerator::paramMinSentenceLength_("minimal-sentence-length", 1, "sentence-generator");

const Core::ParameterInt SentenceGenerator::paramMaxSentenceLength_("maximal-sentence-length", 1, "sentence-generator");

SentenceGenerator::SentenceGenerator() :
		nSentences_(Core::Configuration::config(paramNSentences_)),
		minSentenceLength_(Core::Configuration::config(paramMinSentenceLength_)),
		maxSentenceLength_(Core::Configuration::config(paramMaxSentenceLength_)),
		isInitialized_(false)
{}

void SentenceGenerator::initialize() {
	require_gt(nSentences_, 0);
	require_gt(minSentenceLength_, 0);
	require_le(minSentenceLength_, maxSentenceLength_);
	lm_.loadModel();
	isInitialized_ = true;
}

f32 SentenceGenerator::probability(Word w, const Sentence& history) const {
	switch (lm_.nGramOrder()) {
	case 0:
		return lm_.probability();
	case 1:
		return lm_.probability(w);
	case 2:
		return lm_.probability(w, history.size() > 0 ? history.back() : NGram::senStart);
	case 3:
		return lm_.probability(w, history.size() > 0 ? history.back() : NGram::senStart,
				history.size() > 1 ? history.at(history.size() - 2) : NGram::senStart);
	default:
		return 0;
	}
}

void SentenceGenerator::generateSentence(Sentence& sentence) const {
	sentence.clear();
	Math::RandomNumberGenerator r;
	u32 length = r.randomInt(minSentenceLength_, maxSentenceLength_);
	for (u32 i = 0; i < length; i++) {
		Float p = r.random();
		Float q = 0;
		for (Word w = 0; w < (Word)lm_.lexiconSize(); w++) {
			q += probability(w, sentence);
			if (q >= p) {
				sentence.push_back(w);
				break;
			}
		}
	}
}

void SentenceGenerator::generateSentences(std::vector<Sentence>& sentences) const {
	require(isInitialized_);
	sentences.clear();
	sentences.resize(nSentences_);
	for (u32 i = 0; i < nSentences_; i++) {
		generateSentence(sentences.at(i));
	}
}

void SentenceGenerator::logSentences() const {
	require(isInitialized_);
	std::vector<Sentence> sentences;
	generateSentences(sentences);
	Core::Log::openTag("sentence-generator");
	for (u32 i = 0; i < sentences.size(); i++) {
		require_gt(sentences.at(i).size(), 0);
		std::stringstream s;
		for (u32 j = 0; j < sentences.at(i).size() - 1; j++) {
			s << sentences.at(i).at(j) << " ";
		}
		s << sentences.at(i).back();
		Core::Log::os(s.str().c_str());
	}
	Core::Log::closeTag();
}
