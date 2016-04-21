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

#ifndef NN_MATRIXCONTAINER_HH_
#define NN_MATRIXCONTAINER_HH_

#include <Core/CommonHeaders.hh>
#include <map>
#include "Types.hh"

namespace Nn {

/*
 * container for matrices (may contain matrices/activations for multiple time frames for recurrent neural networks)
 */
class MatrixContainer
{
private:
	std::vector<Matrix*> container_;	// it is most efficient to store pointers to matrices due to resizes of the array
	u32 nTimeframes_;
	u32 maxMemory_;

	bool isComputing_;
public:
	MatrixContainer();
	virtual ~MatrixContainer();

	// set the maximal memory of the container (at most the most recent maxMemory_ matrices are memorized)
	void setMaximalMemory(u32 maxMemory);

	// return number of time frames that are stored
	u32 nTimeframes() { return nTimeframes_; }
	// return the maximal history of time frames that can be stored
	u32 maxMemory() { return maxMemory_; }
	// return the activation for layer l at time frame t
	Matrix& at(u32 timeframe);
	Matrix& getLast();
	void setToZero();
	void reset();

	void addTimeframe(u32 nRows = 0, u32 nColumns = 0);

	bool isComputing() const { return isComputing_; }
	void initComputation(bool sync = true);
	void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_MATRIXCONTAINER_HH_ */
