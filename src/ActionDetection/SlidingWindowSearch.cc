#include "SlidingWindowSearch.hh"
#include <cmath>

using namespace ActionDetection;

const Core::ParameterInt SlidingWindowSearch::paramMinWindowSize_("min-window-size", 0, "sliding-window-search");

const Core::ParameterInt SlidingWindowSearch::paramMaxWindowSize_("max-window-size", 0, "sliding-window-search");

const Core::ParameterInt SlidingWindowSearch::paramInitialStepSize_("initial-step-size", 0, "sliding-window-search");
// join two consecutive detections if they have the same label?
const Core::ParameterBool SlidingWindowSearch::paramJoinConsecutiveActions_("join-consecutive-actions", false, "sliding-window-search");

SlidingWindowSearch::Window::Window(u32 left, u32 right, u32 label, Float score) :
		left(left),
		right(right),
		label(label),
		score(score)
{}

bool SlidingWindowSearch::Window::isOverlapping(const Window& w) const {
		if ((w.left > right) || (w.right < left))
			return false;
		else
			return true;
}

SlidingWindowSearch::SlidingWindowSearch() :
		minWindowSize_(Core::Configuration::config(paramMinWindowSize_)),
		maxWindowSize_(Core::Configuration::config(paramMaxWindowSize_)),
		stepSize_(Core::Configuration::config(paramInitialStepSize_)),
		joinConsecutiveActions_(Core::Configuration::config(paramJoinConsecutiveActions_)),
		scorer_(0),
		isInitialized_(false)
{
	require_le(minWindowSize_, maxWindowSize_);
	require_gt(minWindowSize_, 0);
	require_gt(stepSize_, 0);
	require_ge(minWindowSize_, stepSize_);
}

SlidingWindowSearch::~SlidingWindowSearch() {
	if (scorer_)
		delete scorer_;
}

void SlidingWindowSearch::initialize() {
	scorer_ = Scorer::create();
	scorer_->initialize();
}

void SlidingWindowSearch::search(const Math::Matrix<Float>& sequence) {
	u32 T = sequence.nColumns();
	scorer_->setSequence(sequence);
	std::vector<Window> windows;
	u32 windowSize = minWindowSize_;

	// for all scales
	while (windowSize <= maxWindowSize_) {
		// for all positions
		for (u32 left = 0; left + windowSize - stepSize_ < T; left += stepSize_) {
			u32 right = std::min(left + windowSize - 1, T - 1);
			// get best label and score for this window
			u32 bestLabel = 0;
			Float bestScore = -Types::inf<Float>();
			for (u32 c = 0; c < scorer_->nClasses(); c++) {
				Float score = scorer_->score(c, left, right);
				if (score > bestScore) {
					bestScore = score;
					bestLabel = c;
				}
			}
			windows.push_back(Window(left, right, bestLabel, bestScore));
		}
		// next scale
		windowSize = (u32)(windowSize * std::sqrt(2.0));
		stepSize_ = (u32)(stepSize_ * std::sqrt(2.0));
	}

	// sort windows according to score
	std::sort(windows.begin(), windows.end(), windowConfCompare);
	// remove all overlapping windows
	for (u32 i = 0; i < windows.size(); i++) {
		for (u32 j = i+1; j < windows.size(); j++) {
			if (windows.at(i).isOverlapping(windows.at(j))) {
				windows.erase(windows.begin() + j);
				j--;
			}
		}
	}
	// sort windows according to starting time
	std::sort(windows.begin(), windows.end(), windowPosCompare);

	// make sure first window starts at position 0
	windows.front().left = 0;

	if (joinConsecutiveActions_) {
		for (u32 i = 1; i < windows.size(); i++) {
			if (windows.at(i-1).label == windows.at(i).label) {
				Float scaleW1 = (Float)(windows.at(i-1).right - windows.at(i-1).left);
				Float scaleW2 = (Float)(windows.at(i).right - windows.at(i).left);
				scaleW1 /= scaleW1 + scaleW2;
				scaleW2 /= scaleW1 + scaleW2;
				windows.at(i-1).right = windows.at(i).right;
				windows.at(i-1).score = scaleW1 * windows.at(i-1).score + scaleW2 * windows.at(i).score;
				windows.erase(windows.begin() + i);
				i--;
			}
		}
	}

	// log the resulting sequence
	Core::Log::openTag("best-sequence");
	std::stringstream s;
	Float score = 0;
	for (u32 i = 0; i < windows.size(); i++) {
		s << windows.at(i).label << ":" << windows.at(i).left << ":" << windows.at(i).score << " ";
		score += windows.at(i).score;
	}
	Core::Log::os() << s.str();
	Core::Log::os("score: ") << score;
	Core::Log::closeTag();
}
