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

#ifndef ACTIONDETECTION_SLIDINGWINDOWSEARCH_HH_
#define ACTIONDETECTION_SLIDINGWINDOWSEARCH_HH_

#include <Core/CommonHeaders.hh>
#include "Scorer.hh"
#include <Math/Matrix.hh>
#include <algorithm>

namespace ActionDetection {

/*
 * compute scores for each action on a gradually increasing sliding window
 * non-maximum suppression is applied to obtain a segmentation of the input into multiple action segments of different classes
 */
class SlidingWindowSearch
{
private:
	// window class
	class Window {
	public:
		u32 left;
		u32 right;
		u32 label;
		Float score;
		Window(u32 left, u32 right, u32 label, Float score);
		bool isOverlapping(const Window& w) const;
	};

	// comparator to sort windows in descending order according to their scores
	static bool windowConfCompare(const Window& w1, const Window& w2) { return w1.score > w2.score; }
	// comparator to sort windows in ascending order according to their left positions
	static bool windowPosCompare(const Window& w1, const Window& w2) { return w1.left < w2.left; }

	static const Core::ParameterInt paramMinWindowSize_;
	static const Core::ParameterInt paramMaxWindowSize_;
	static const Core::ParameterInt paramInitialStepSize_;
	static const Core::ParameterBool paramJoinConsecutiveActions_;

	u32 minWindowSize_;
	u32 maxWindowSize_;
	u32 stepSize_;
	bool joinConsecutiveActions_;
	Scorer* scorer_;
	bool isInitialized_;
public:
	SlidingWindowSearch();
	virtual ~SlidingWindowSearch();
	void initialize();
	/*
	 * @param sequence the sequence of input feature vectors used to infer an temporally localized action classes
	 */
	void search(const Math::Matrix<Float>& sequence);
};

} // namespace

#endif /* ACTIONDETECTION_SLIDINGWINDOWSEARCH_HH_ */
