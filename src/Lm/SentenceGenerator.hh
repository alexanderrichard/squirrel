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

#ifndef LM_SENTENCEGENERATOR_HH_
#define LM_SENTENCEGENERATOR_HH_

#include "Core/CommonHeaders.hh"
#include "LanguageModel.hh"

namespace Lm {

/*
 * draw sentences from the language model
 */
class SentenceGenerator
{
public:
	typedef std::vector<Lm::Word> Sentence;
private:
	static const Core::ParameterInt paramNSentences_;
	static const Core::ParameterInt paramMinSentenceLength_;
	static const Core::ParameterInt paramMaxSentenceLength_;
	u32 nSentences_;
	u32 minSentenceLength_;
	u32 maxSentenceLength_;
	NGram lm_;
	bool isInitialized_;
private:
	f32 probability(Word w, const Sentence& history) const;
	void generateSentence(Sentence& sentence) const;
public:
	SentenceGenerator();
	virtual ~SentenceGenerator() {}
	void initialize();
	/*
	 * @param sentences contains the generated sentences after the function terminates
	 */
	void generateSentences(std::vector<Sentence>& sentences) const;
	/*
	 * generate sentences and write them directly to the log-file
	 */
	void logSentences() const;
};

} // namespace

#endif /* LM_SENTENCEGENERATOR_HH_ */
