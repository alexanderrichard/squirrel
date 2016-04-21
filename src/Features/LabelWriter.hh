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

#ifndef LABELWRITER_HH_
#define LABELWRITER_HH_

#include "FeatureWriter.hh"
#include <vector>

namespace Features {

class LabelWriter : public FeatureWriter
{
private:
	typedef FeatureWriter Precursor;
private:
	// make these methods inaccessible from outside
	virtual void write(const Math::Matrix<Float>& featureSequence);
	virtual void write(const Math::Vector<Float>& feature);
public:
	LabelWriter(const char* name = "features.label-writer");
	virtual ~LabelWriter() {}

	void write(u32 label);
	void write(const std::vector<u32>& labelSequence);
};

class SequenceLabelWriter : public SequenceFeatureWriter
{
private:
	typedef SequenceFeatureWriter Precursor;
private:
	// make these methods inaccessible from outside
	virtual void write(std::vector<u32>& timestamps, const Math::Matrix<Float>& featureSequence);
	virtual void write(const Math::Matrix<Float>& featureSequence);
public:
	SequenceLabelWriter(const char* name = "features.label-writer");
	virtual ~SequenceLabelWriter() {}

	/* writing routines, pass labels (and optionally timestamps for each label) to these functions */
	void write(const std::vector<u32>& labelSequence);
	void write(std::vector<u32>& timestamps, const std::vector<u32>& labelSequence);
};

} // namespace

#endif /* LABELWRITER_HH_ */
