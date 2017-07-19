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
#include "MultiChannelRbfChiSquareKernel.hh"
#include "Normalizations.hh"
#include "PrincipalComponentAnalysis.hh"
#include <Math/Matrix.hh>
#include <Features/FeatureReader.hh>
#include <Features/FeatureWriter.hh>
#include <iostream>

using namespace FeatureTransformation;

APPLICATION(FeatureTransformation::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, kernel-transformation, mean-and-variance-estimation, min-max-estimation, principal-component-analysis",
		"none");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case kernelTransformation:
		{
		Kernel* kernel = Kernel::createKernel();
		kernel->initialize();
		kernel->applyKernel();
		kernel->finalize();
		delete kernel;
		}
		break;
	case meanAndVarianceEstimation:
		{
		MeanAndVarianceEstimation normalizer;
		normalizer.estimate();
		}
		break;
	case minMaxEstimation:
		{
		MinMaxEstimation normalizer;
		normalizer.estimate();
		}
		break;
	case principalComponentAnalysis:
		{
		PrincipalComponentAnalysis pca;
		pca.initialize();
		pca.estimate();
		}
		break;
	case none:
	default:
		std::cerr << "No action given. Abort." << std::endl;
		exit(1);
	}
}
