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

#ifndef ACTIONDETECTION_LINEARSEARCH_HH_
#define ACTIONDETECTION_LINEARSEARCH_HH_

#include <Core/CommonHeaders.hh>
#include <Lm/LanguageModel.hh>
#include "LengthModelling.hh"
#include "Scorer.hh"
#include <Math/Matrix.hh>

namespace ActionDetection {

/*
 * Viterbi algorithm to search for best action classes and locations
 */
class LinearSearch
{
private:
	static const Core::ParameterFloat paramLengthModelScale_;
	static const Core::ParameterFloat paramLanguageModelScale_;
	static const Core::ParameterInt paramTemporalStride_;
	static const Core::ParameterInt paramMaximalActionLength_;
	static const Core::ParameterBool paramScorePerFrame_;

	Float lengthModelScale_;
	Float languageModelScale_;
	u32 temporalStride_;
	u32 maximalActionLength_;
	Lm::NGram languageModel_;
	LengthModel* lengthModel_;
	Scorer* scorer_;
	bool scorePerFrame_;
	std::vector< Math::Matrix<Float> > lmScores_; // contains precomputed lm-scores
	bool isInitialized_;

	Float lmScore(Lm::Word w, Lm::Word h);
	Float lmScoreTrigram(Lm::Word w, Lm::Word h, Lm::Word hh);
	Float lengthScore(u32 l, Lm::Word c);
	Float actionScore(u32 c, u32 t_start, u32 t_end);
	void _viterbiDecoding(const Math::Matrix<Float>& sequence);
	void _viterbiDecodingWithTrigram(const Math::Matrix<Float>& sequence);
public:
	LinearSearch();
	virtual ~LinearSearch();
	/*
	 * initialize the search algorithm
	 */
	void initialize();
	/*
	 *  @param sequence the sequence of input feature vectors used to infer an temporally localized action classes
	 */
	void viterbiDecoding(const Math::Matrix<Float>& sequence);
};

} // namespace

#endif /* ACTIONDETECTION_LINEARSEARCH_HH_ */
