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

#include "LabelConversion.hh"

using namespace Converter;

/*
 * AsciiLabelConverter
 */

const Core::ParameterString AsciiLabelConverter::paramAsciiLabelFile_("label-file", "", "converter.label-converter");

const Core::ParameterBool AsciiLabelConverter::paramIsSequenceLabelFile_("is-sequence-label-file", false, "converter.label-converter");

AsciiLabelConverter::AsciiLabelConverter() :
		asciiLabelFile_(Core::Configuration::config(paramAsciiLabelFile_)),
		isSequenceLabelFile_(Core::Configuration::config(paramIsSequenceLabelFile_))
{}

void AsciiLabelConverter::_writeLabelCache() {
	Features::LabelWriter labelWriter;
	Core::AsciiStream labelsIn(asciiLabelFile_, std::ios::in);
	u32 label;
	labelsIn >> label;
	while (!labelsIn.eof()) {
		labelWriter.write(label);
		labelsIn >> label;
	}
	labelWriter.finalize();
}

void AsciiLabelConverter::_writeSequenceLabelCache() {
	Features::SequenceLabelWriter labelWriter;
	Core::AsciiStream labelsIn(asciiLabelFile_, std::ios::in);
	u32 timestamp;
	u32 label;
	std::vector<u32> timestamps;
	std::vector<u32> labels;
	labelsIn >> timestamp;
	labelsIn >> label;
	while (!labelsIn.eof()) {
		timestamps.push_back(timestamp);
		labels.push_back(label);
		labelsIn >> timestamp;
		labelsIn >> label;
	}
	labelWriter.write(timestamps, labels);
	labelWriter.finalize();
}

void AsciiLabelConverter::writeLabelCache() {
	if (isSequenceLabelFile_)
		_writeSequenceLabelCache();
	else
		_writeLabelCache();
}

/*
 * SingleLabelToSequenceLabelConverter
 */

SingleLabelToSequenceLabelConverter::SingleLabelToSequenceLabelConverter() :
		featureReader_("converter.label-converter.sequence-feature-reader"),
		labelReader_("converter.label-converter.label-reader"),
		labelWriter_("converter.label-converter.label-writer")
{}

void SingleLabelToSequenceLabelConverter::convert() {
	featureReader_.initialize();
	labelReader_.initialize();
	require(featureReader_.totalNumberOfSequences() == labelReader_.totalNumberOfFeatures());
	while (featureReader_.hasSequences()) {
		u32 label = labelReader_.next();
		u32 sequenceLength = featureReader_.next().nColumns();
		std::vector<u32> labels(sequenceLength, label);
		labelWriter_.write(featureReader_.currentTimestamps(), labels);
	}
	labelWriter_.finalize();
}
