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
 * Application.cc
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#include "Application.hh"
#include "GrammarGenerator.hh"
#include "ViterbiDecoding.hh"
#include <Features/FeatureReader.hh>
#include <Features/FeatureWriter.hh>
#include <iostream>
#include <sstream>

using namespace Hmm;

APPLICATION(Hmm::Application)

const Core::ParameterEnum Application::paramAction_("action", "none, generate-grammar, viterbi-decoding, realignment", "none");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case generateGrammar:
		{
		GrammarGenerator* g = GrammarGenerator::create();
		g->generate();
		delete g;
		}
		break;
	case viterbiDecoding:
		decode();
		break;
	case realignment:
		realign();
		break;
	case none:
	default:
		Core::Error::msg("No action given.") << Core::Error::abort;
	}
}

void Application::decode() {
	ViterbiDecoding v;
	v.initialize();
	v.sanityCheck();

	Features::SequenceFeatureReader reader;
	reader.initialize();
	Features::SequenceLabelWriter labelWriter;
	labelWriter.initialize(reader.totalNumberOfFeatures(), v.nOutputClasses(), reader.totalNumberOfSequences());

	while (reader.hasSequences()) {
		Float score = v.decode(reader.next());
		Core::Log::openTag("sequence");
		Core::Log::openTag("score");
		Core::Log::os() << score;
		Core::Log::closeTag();
		Core::Log::openTag("recognized");
		std::stringstream s;
		for (u32 i = 0; i < v.segmentation().size(); i++)
			s << v.segmentation().at(i).label << ":" << v.segmentation().at(i).length << " ";
		Core::Log::os(s.str().c_str());
		Core::Log::closeTag();
		Core::Log::closeTag();
		labelWriter.write(v.framewiseRecognition());
	}

	labelWriter.finalize();
}

void Application::realign() {
	ViterbiDecoding v;
	v.initialize();

	Features::SequenceFeatureReader reader;
	Features::SequenceLabelReader labelReader;
	reader.initialize();
	labelReader.initialize();
	if (reader.totalNumberOfSequences() != labelReader.totalNumberOfSequences())
		Core::Error::msg("Application::realign: features.feature-reader and features.label-reader need to have the same number of sequence.") << Core::Error::abort;
	Features::SequenceLabelWriter labelWriter;
	labelWriter.initialize(reader.totalNumberOfFeatures(), v.nOutputClasses(), reader.totalNumberOfSequences());

	while (reader.hasSequences()) {
		Float score = v.realign(reader.next(), labelReader.nextLabelSequence());
		Core::Log::openTag("sequence");
		Core::Log::openTag("score");
		Core::Log::os() << score;
		Core::Log::closeTag();
		Core::Log::openTag("recognized");
		std::stringstream s;
		for (u32 i = 0; i < v.segmentation().size(); i++)
			s << v.segmentation().at(i).label << ":" << v.segmentation().at(i).length << " ";
		Core::Log::os(s.str().c_str());
		Core::Log::closeTag();
		Core::Log::closeTag();
		labelWriter.write(v.framewiseRecognition());
	}

	labelWriter.finalize();
}
