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
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#include "Application.hh"
#include "FeatureCacheManager.hh"
#include <iostream>

using namespace Features;

APPLICATION(Features::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, print-cache",
		"none");

void Application::main() {
	switch (Core::Configuration::config(paramAction_)) {
	case printCache:
		{
		FeatureCachePrinter printer;
		printer.work();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}
