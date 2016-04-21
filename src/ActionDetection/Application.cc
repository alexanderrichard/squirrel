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

#include "Application.hh"
#include "LinearSearch.hh"
#include "SlidingWindowSearch.hh"
#include <Features/FeatureReader.hh>
#include <iostream>
#include "Scorer.hh"
#include <sstream>

using namespace ActionDetection;

APPLICATION(ActionDetection::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, linear-search, sliding-window-search, estimate-poisson-model", "none");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case linearSearch:
		{
		Features::SequenceFeatureReader featureReader;
		LinearSearch linearSearch;
		featureReader.initialize();
		linearSearch.initialize();
		while (featureReader.hasSequences()) {
			linearSearch.viterbiDecoding(featureReader.next());
		}
		}
		break;
	case slidingWindowSearch:
		{
		Features::SequenceFeatureReader featureReader;
		SlidingWindowSearch slidingWindowSearch;
		featureReader.initialize();
		slidingWindowSearch.initialize();
		while (featureReader.hasSequences()) {
			slidingWindowSearch.search(featureReader.next());
		}
		}
		break;
	case estimatePoissonModel:
		{
		PoissonModel l;
		l.estimate();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}
