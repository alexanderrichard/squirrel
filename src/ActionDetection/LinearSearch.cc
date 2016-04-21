#include "LinearSearch.hh"
#include <sstream>

using namespace ActionDetection;

const Core::ParameterFloat LinearSearch::paramLengthModelScale_("length-model-scale", 1.0, "linear-search");

const Core::ParameterFloat LinearSearch::paramLanguageModelScale_("language-model-scale", 1.0, "linear-search");

const Core::ParameterInt LinearSearch::paramTemporalStride_("temporal-stride", 1, "linear-search");

const Core::ParameterInt LinearSearch::paramMaximalActionLength_("maximal-action-length", Types::max<u32>(), "linear-search");

const Core::ParameterBool LinearSearch::paramScorePerFrame_("score-per-frame", true, "linear-search");

LinearSearch::LinearSearch() :
		lengthModelScale_(Core::Configuration::config(paramLengthModelScale_)),
		languageModelScale_(Core::Configuration::config(paramLanguageModelScale_)),
		temporalStride_(Core::Configuration::config(paramTemporalStride_)),
		maximalActionLength_(Core::Configuration::config(paramMaximalActionLength_)),
		lengthModel_(0),
		scorer_(0),
		scorePerFrame_(Core::Configuration::config(paramScorePerFrame_)),
		isInitialized_(false)
{
	require_gt(temporalStride_, 0);
	maximalActionLength_ /= temporalStride_;
}

LinearSearch::~LinearSearch() {
	if (lengthModel_)
		delete lengthModel_;
	if (scorer_)
		delete scorer_;
}

void LinearSearch::initialize() {
	languageModel_.loadModel();
	require_le(languageModel_.nGramOrder(), 3);

	lengthModel_ = LengthModel::create();
	lengthModel_->initialize();

	scorer_ = Scorer::create();
	scorer_->initialize();

	isInitialized_ = true;
}

Float LinearSearch::lmScore(Lm::Word w, Lm::Word h) {
	require(isInitialized_);
	// if language model scale is 0, return 0
	if (languageModelScale_ == 0)
		return 0;

	// precompute lm-scores if not yet done
	if (lmScores_.size() == 0) {
		lmScores_.resize(1);
		lmScores_.at(0).resize(languageModel_.lexiconSize(), languageModel_.lexiconSize() + 1);
		lmScores_.at(0).fill(-Types::inf<Float>());
		// run over all words and predecessor words
		for (Lm::Word w = 0; w < (Lm::Word)languageModel_.lexiconSize(); w++) {
			for (Lm::Word h = 0; h < (Lm::Word)languageModel_.lexiconSize(); h++) {
				if (languageModel_.nGramOrder() == 0)
					lmScores_.at(0).at((u32)w, (u32)h) = languageModelScale_ * std::log(languageModel_.probability());
				else if (languageModel_.nGramOrder() == 1)
					lmScores_.at(0).at((u32)w, (u32)h) = languageModelScale_ * std::log(languageModel_.probability(w));
				else
					lmScores_.at(0).at((u32)w, (u32)h) = languageModelScale_ * std::log(languageModel_.probability(w, h));
			}
			// if h is sentence start symbol
			if (languageModel_.nGramOrder() == 0)
				lmScores_.at(0).at((u32)w, languageModel_.lexiconSize()) = languageModelScale_ * std::log(languageModel_.probability());
			else if (languageModel_.nGramOrder() == 1)
				lmScores_.at(0).at((u32)w, languageModel_.lexiconSize()) = languageModelScale_ * std::log(languageModel_.probability(w));
			else
				lmScores_.at(0).at((u32)w, languageModel_.lexiconSize()) = languageModelScale_ * std::log(languageModel_.probability(w, Lm::NGram::senStart));
		}
	}

	if (h == Lm::NGram::senStart)
		return lmScores_.at(0).at((u32)w, languageModel_.lexiconSize());
	else
		return lmScores_.at(0).at((u32)w, (u32)h);
}

Float LinearSearch::lmScoreTrigram(Lm::Word w, Lm::Word h, Lm::Word hh) {
	require(isInitialized_);
	// if language model scale is 0, return 0
	if (languageModelScale_ == 0)
		return 0;

	// precompute lm-scores if not yet done
	if (lmScores_.size() == 0) {
		lmScores_.resize(languageModel_.lexiconSize());
		for (u32 w = 0; w < languageModel_.lexiconSize(); w++) {
			lmScores_.at(w).resize(languageModel_.lexiconSize() + 1, languageModel_.lexiconSize() + 1);
			lmScores_.at(w).fill(-Types::inf<Float>());
			// run over all predecessor words and prepredecessor words
			for (Lm::Word h = 0; h < (Lm::Word)languageModel_.lexiconSize(); h++) {
				for (Lm::Word hh = 0; hh < (Lm::Word)languageModel_.lexiconSize(); hh++) {
					lmScores_.at(w).at((u32)h, (u32)hh) = std::log(languageModel_.probability(w, h, hh));
				}
				// if hh is sentence start symbol
				lmScores_.at(w).at((u32)h, languageModel_.lexiconSize()) = std::log(languageModel_.probability(w, h, Lm::NGram::senStart));
			}
			// if h is sentence start symbol
			lmScores_.at(w).at(languageModel_.lexiconSize(), languageModel_.lexiconSize()) = std::log(languageModel_.probability(w, Lm::NGram::senStart, Lm::NGram::senStart));
		}
	}

	return languageModelScale_ * lmScores_.at(w).at((u32)h, (u32)hh);
}

Float LinearSearch::lengthScore(u32 l, Lm::Word c) {
	require(isInitialized_);

	l *= temporalStride_;

	// if length model scale is 0, return 0
	if (lengthModelScale_ == 0)
		return 0;

	return lengthModelScale_ * lengthModel_->logProbability(l, (u32)c);
}

Float LinearSearch::actionScore(u32 c, u32 t_start, u32 t_end) {
	require_le(t_start, t_end);
	Float score = scorer_->score(c, t_start, t_end);
	if (scorePerFrame_)
		score *= (t_end - t_start + 1) * temporalStride_;
	return score;
}

void LinearSearch::_viterbiDecoding(const Math::Matrix<Float>& sequence) {
	require(isInitialized_);
	u32 T = sequence.nColumns();
	u32 nClasses = languageModel_.lexiconSize();

	scorer_->setSequence(sequence);

	Math::Matrix<Float> Q(nClasses, T);
	Q.fill(-Types::inf<Float>());
	Math::Matrix<Float> C(nClasses, T); // best predecessor class of class c ending at time t
	C.fill(Lm::NGram::senStart);
	Math::Matrix<Float> L(nClasses, T); // starting frame of class c ending at time t
	L.fill(1);

	/* run over all timeframes */
	for (u32 t = 0; t < T; t++) {
		/* run over all possible action lengths */
		u32 h = std::min(t + 1, maximalActionLength_);
		for (u32 l = h; l > 0; l--) {
			/* run over all classes */
			for (Lm::Word c = 0; c < (Lm::Word)nClasses; c++) {
				Float score;
				Float action_score = actionScore((u32)c, t+1-l, t);
				Float length_score = lengthScore(l, c);
				if (l > t) { // first action in sequence
					score = lmScore(c, Lm::NGram::senStart) + length_score + action_score;
					if (Q.at((u32)c, t) < score) {
						Q.at((u32)c, t) = score;
						C.at((u32)c, t) = Lm::NGram::senStart;
						L.at((u32)c, t) = 0;
					}
				}
				else { // there is a preceding action
					/* run over all possible predecessor classes */
					for (Lm::Word cc = 0; cc < (Lm::Word)nClasses; cc++) {
						Float score = Q.at((u32)cc, t-l) + length_score + lmScore(c, cc);
						score += action_score;
						if (Q.at((u32)c, t) < score) {
							Q.at((u32)c, t) = score;
							C.at((u32)c, t) = cc;
							L.at((u32)c, t) = t - l + 1;
						}
					}
				}
			}
		}
	}

	/* find best ending class */
	Lm::Word bestEndingClass = 0;
	Float bestEndingScore = Q.at(0, T-1);
	u32 bestEndingFrame = T-1;
	for (Lm::Word c = 1; c < (Lm::Word)nClasses; c++) {
		if (Q.at(c, T-1) > bestEndingScore) {
			bestEndingClass = c;
			bestEndingScore = Q.at(c, T-1);
		}
	}
	/* reconstruct best path through the sequence */
	std::vector<Lm::Word> classSequence;
	std::vector<u32> startFrames;
	while (bestEndingClass != Lm::NGram::senStart) {
		classSequence.insert(classSequence.begin(), bestEndingClass);
		startFrames.insert(startFrames.begin(), L.at((u32)bestEndingClass, bestEndingFrame));
		bestEndingClass = C.at((u32)bestEndingClass, bestEndingFrame);
		bestEndingFrame = startFrames.front() - 1;
	}

	/* log the reconstructed sequence */
	Core::Log::openTag("best-sequence");
	std::stringstream s;
	scorePerFrame_ = false; // confidence score on segment level
	for (u32 i = 0; i < classSequence.size(); i++) {
		u32 startFrame = startFrames.at(i);
		u32 endFrame = (i+1 < classSequence.size() ? startFrames.at(i+1) : T) - 1;
		Float score = actionScore((u32)classSequence.at(i), startFrame, endFrame);
		s << classSequence.at(i) << ":" << startFrames.at(i) * temporalStride_ << ":" << score << " ";
	}
	Core::Log::os() << s.str();
	Core::Log::os("score: ") << bestEndingScore;
	Core::Log::closeTag();
}

void LinearSearch::_viterbiDecodingWithTrigram(const Math::Matrix<Float>& sequence) {
	require(isInitialized_);
	u32 T = sequence.nColumns();
	u32 nClasses = languageModel_.lexiconSize();

	scorer_->setSequence(sequence);

	std::vector< Math::Matrix<Float> > Q(nClasses+1); // +1 for senStart
	std::vector< Math::Matrix<Float> > C(nClasses+1);
	std::vector< Math::Matrix<Float> > L(nClasses+1);
	for (u32 c = 0; c < nClasses+1; c++) {
		Q.at(c).resize(nClasses, T);
		Q.at(c).fill(-Types::inf<Float>());
		C.at(c).resize(nClasses, T);
		C.at(c).fill((Lm::Word)nClasses);
		L.at(c).resize(nClasses, T);
		L.at(c).fill(1);
	}

	/* run over all timeframes */
	for (u32 t = 0; t < T; t++) {
		/* run over all possible action lengths */
		u32 h = std::min(t + 1, maximalActionLength_);
		for (u32 l = h; l > 0; l--) {
			/* run over all classes */
			for (Lm::Word c = 0; c < (Lm::Word)nClasses; c++) {
				Float score;
				Float action_score = actionScore((u32)c, t+1-l, t);
				Float length_score = lengthScore(l, c);
				if (l > t) { // first action in sequence
					score = lmScoreTrigram(c, (Lm::Word)nClasses, (Lm::Word)nClasses) + length_score + action_score;
					if (Q.at(nClasses).at((u32)c, t) < score) {
						Q.at(nClasses).at((u32)c, t) = score;
						C.at(nClasses).at((u32)c, t) = (Lm::Word)nClasses; // senStart
						L.at(nClasses).at((u32)c, t) = 0;
					}
				}
				else {
					/* run over all possible predecessor classes */
					for (Lm::Word cc = 0; cc < (Lm::Word)nClasses; cc++) {
						/* run over all possible prepredecessor classes */
						for (Lm::Word ccc = 0; ccc < (Lm::Word)nClasses+1; ccc++) {
							Float score = Q.at(ccc).at((u32)cc, t-l) + length_score + lmScoreTrigram(c, cc, ccc);
							score += action_score;
							if (Q.at(cc).at((u32)c, t) < score) {
								Q.at(cc).at((u32)c, t) = score;
								C.at(cc).at((u32)c, t) = ccc;
								L.at(cc).at((u32)c, t) = t - l + 1;
							}
						}
					}
				}
			}
		}
	}

	/* find best ending class */
	Lm::Word bestEndingClass = 0;
	Lm::Word bestPrecedingClass = 0;
	Float bestEndingScore = Q.at(0).at(0, T-1);
	u32 bestEndingFrame = T-1;
	for (Lm::Word c = 0; c < (Lm::Word)nClasses; c++) {
		for (Lm::Word cc = 0; cc < (Lm::Word)nClasses+1; cc++) {
			if (Q.at(cc).at(c, T-1) > bestEndingScore) {
				bestEndingClass = c;
				bestPrecedingClass = cc;
				bestEndingScore = Q.at(cc).at(c, T-1);
			}
		}
	}

	/* reconstruct best path through the sequence */
	std::vector<Lm::Word> classSequence;
	std::vector<u32> startFrames;
	while (bestEndingClass != (Lm::Word)nClasses) {
		classSequence.insert(classSequence.begin(), bestEndingClass);
		startFrames.insert(startFrames.begin(), L.at(bestPrecedingClass).at((u32)bestEndingClass, bestEndingFrame));
		Lm::Word tmp = bestPrecedingClass;
		bestPrecedingClass = C.at(bestPrecedingClass).at(bestEndingClass, bestEndingFrame);
		bestEndingClass = tmp;
		bestEndingFrame = startFrames.front() - 1;
	}

	/* log the reconstructed sequence */
	Core::Log::openTag("best-sequence");
	std::stringstream s;
	scorePerFrame_ = false; // confidence score on segment level
	for (u32 i = 0; i < classSequence.size(); i++) {
		u32 startFrame = startFrames.at(i);
		u32 endFrame = (i+1 < classSequence.size() ? startFrames.at(i+1) : T) - 1;
		Float score = actionScore((u32)classSequence.at(i), startFrame, endFrame);
		s << classSequence.at(i) << ":" << startFrames.at(i) * temporalStride_ << ":" << score << " ";
	}
	Core::Log::os() << s.str();
	Core::Log::os("score: ") << bestEndingScore;
	Core::Log::closeTag();
}

void LinearSearch::viterbiDecoding(const Math::Matrix<Float>& sequence) {
	if (languageModel_.nGramOrder() < 3) {
		_viterbiDecoding(sequence);
	}
	else {
		_viterbiDecodingWithTrigram(sequence);
	}
}
