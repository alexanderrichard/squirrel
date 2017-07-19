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
 *  Created on: Apr 16, 2014
 *      Author: richard
 */

#include "Application.hh"
#include "LibSvmConversion.hh"
#include "LogLinearConverter.hh"
#include "FileFormatConverter.hh"
#include <iostream>

using namespace Converter;

APPLICATION(Converter::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, convert-file-format, gaussian-mixture-to-log-linear, kmeans-to-log-linear, lib-svm-to-log-linear",
		"none");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case convertFileFormat:
		{
		FileFormatConverter converter;
		converter.convert();
		}
		break;
	case gaussianMixtureToLogLinear:
		{
		GaussianMixtureToLogLinear converter;
		converter.convert();
		}
		break;
	case kMeansToLogLinear:
		{
		KMeansToLogLinear converter;
		converter.convert();
		}
		break;
	case libSvmToLogLinear:
		{
		LibSvmToLogLinear converter;
		converter.convert();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}
