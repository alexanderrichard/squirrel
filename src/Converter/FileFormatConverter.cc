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
 * FileFormatConverter.cc
 *
 *  Created on: May 5, 2017
 *      Author: richard
 */

#include "FileFormatConverter.hh"
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include "Features/FeatureReader.hh"
#include "Features/FeatureWriter.hh"

using namespace Converter;

const Core::ParameterEnum FileFormatConverter::paramFileType_("file-type", "vector, matrix, cache", "cache", "converter");

const Core::ParameterString FileFormatConverter::paramInputFile_("input", "", "converter");

const Core::ParameterString FileFormatConverter::paramOutputFile_("output", "", "converter");

FileFormatConverter::FileFormatConverter() :
		input_(Core::Configuration::config(paramInputFile_)),
		output_(Core::Configuration::config(paramOutputFile_)),
		type_((FileType)Core::Configuration::config(paramFileType_))
{
	if (input_.empty())
		Core::Error::msg("converter.input not given.") << Core::Error::abort;
	if (output_.empty())
		Core::Error::msg("converter.output not given.") << Core::Error::abort;
}

void FileFormatConverter::convertVector() {
	Math::Vector<Float> in;
	in.read(input_);
	in.write(output_);
}

void FileFormatConverter::convertMatrix() {
	Math::Matrix<Float> in;
	in.read(input_);
	in.write(output_);
}

void FileFormatConverter::convertCache() {
	Features::FeatureCache::FeatureType type = Features::FeatureCache::featureType(input_);
	switch (type) {
	case Features::FeatureCache::labels:
	{
		Features::LabelReader reader("reader", input_, 1, false);
		reader.initialize();
		Features::LabelWriter writer("writer", output_);
		writer.initialize(reader.totalNumberOfFeatures(), reader.featureDimension());
		while (reader.hasFeatures())
			writer.write(reader.nextLabel());
	}
	break;
	case Features::FeatureCache::sequencelabels:
	{
		Features::SequenceLabelReader reader("reader", input_, 1, false, false);
		reader.initialize();
		Features::SequenceLabelWriter writer("writer", output_);
		writer.initialize(reader.totalNumberOfFeatures(), reader.featureDimension(), reader.totalNumberOfSequences());
		while (reader.hasSequences())
			writer.write(reader.nextLabelSequence());
	}
	break;
	case Features::FeatureCache::vectors:
	{
		Features::FeatureReader reader("reader", input_, 1, false);
		reader.initialize();
		Features::FeatureWriter writer("writer", output_);
		writer.initialize(reader.totalNumberOfFeatures(), reader.featureDimension());
		while (reader.hasFeatures())
			writer.write(reader.next());
	}
	break;
	case Features::FeatureCache::sequences:
	{
		Features::SequenceFeatureReader reader("reader", input_, 1, false, false);
		reader.initialize();
		Features::SequenceFeatureWriter writer("writer", output_);
		writer.initialize(reader.totalNumberOfFeatures(), reader.featureDimension(), reader.totalNumberOfSequences());
		while (reader.hasSequences())
			writer.write(reader.next());
	}
	break;
	default:
		Core::Error::msg("FileFormatConverter::convertCache: only feature caches of type "
				"#labels, #sequencelabels, #vectors, and #sequences can be converted.") << Core::Error::abort;
	}
}

void FileFormatConverter::convert() {
	switch (type_) {
	case vector:
		convertVector();
		break;
	case matrix:
		convertMatrix();
		break;
	case cache:
		convertCache();
		break;
	default:
		;
	}
}
