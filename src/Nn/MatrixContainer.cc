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

/*
 * MatrixContainer.cc
 * MatrixContainer.cc
 *
 *  Created on: May 22, 2014
 *      Author: richard
 */

#include "MatrixContainer.hh"

using namespace Nn;

MatrixContainer::MatrixContainer() :
		nTimeframes_(0),
		maxMemory_(0),
		isComputing_(false)
{}

MatrixContainer::~MatrixContainer() {
	for (u32 i = 0; i < container_.size(); i++) {
		if (container_.at(i))
			delete container_.at(i);
	}
}

void MatrixContainer::setMaximalMemory(u32 maxMemory) {
	require_gt(maxMemory, 0);
	maxMemory_ = maxMemory;
	// ensure sufficient size of container
	if (container_.size() < maxMemory_) {
		u32 oldSize = container_.size();
		// create the new elements
		for (u32 i = oldSize; i < maxMemory_; i++) {
			container_.push_back(new Matrix);
		}
	}
	// update the computing state of each activation
	if (isComputing_) {
		initComputation(false);
	}
	else {
		finishComputation(false);
	}
}

Matrix& MatrixContainer::at(u32 timeframe) {
	require_lt(timeframe, nTimeframes_);
	u32 oldestMemorizedTimeframe = std::max(0, (s32)nTimeframes_ - (s32)maxMemory_);
	if (timeframe < oldestMemorizedTimeframe) {
		std::cerr << "MatrixContainer: Cannot access timeframe " << timeframe << ", only timeframes "
				<< oldestMemorizedTimeframe << ",...," << nTimeframes_ - 1
				<< " are memorized. Abort." << std::endl;
		exit(1);
	}
	return *(container_.at(timeframe - oldestMemorizedTimeframe));
}

Matrix& MatrixContainer::getLast() {
	require_gt(nTimeframes_, 0);
	return at(nTimeframes_ - 1);
}

void MatrixContainer::setToZero() {
	require_le(maxMemory_, container_.size());
	for (u32 i = 0; i < maxMemory_; i++) {
		container_.at(i)->setToZero();
	}
}

void MatrixContainer::reset() {
	nTimeframes_ = 0;
}

void MatrixContainer::addTimeframe(u32 nRows, u32 nColumns) {
	require(maxMemory_ > 0);
	nTimeframes_++;
	if (nTimeframes_ > maxMemory_) {
		// forget oldest time frame and make space for a new time frame
		for (u32 i = 1; i < maxMemory_; i++) {
			container_.at(i)->swap(*(container_.at(i-1)));
		}
	}
	getLast().resize(nRows, nColumns);
}

void MatrixContainer::initComputation(bool sync) {
	require_le(maxMemory_, container_.size());
	for (u32 t = 0; t < maxMemory_; t++) {
		container_.at(t)->initComputation(sync);
	}
	isComputing_ = true;
}

void MatrixContainer::finishComputation(bool sync) {
	require_le(maxMemory_, container_.size());
	for (u32 t = 0; t < maxMemory_; t++) {
		container_.at(t)->finishComputation(sync);
	}
	isComputing_ = false;
}
