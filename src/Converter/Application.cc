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
#include "DenseTrajectoriesConversion.hh"
#include "LabelConversion.hh"
#include "LibSvmConversion.hh"
#include "MatrixToCacheConversion.hh"
#include "AsciiCacheToBinaryCache.hh"
#include <iostream>

using namespace Converter;

APPLICATION(Converter::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"none, internal-cache-conversion, external-to-cache, cache-to-external, lib-svm-to-log-linear",
		"none");

const Core::ParameterEnum Application::paramExternalType_("external-type",
		"ascii-feature-cache, ascii-labels, dense-trajectories, lib-svm", "ascii-feature-cache");

const Core::ParameterEnum Application::paramInternalCacheConversions_("internal-cache-conversion-type",
		"single-label-to-sequence-label, matrix-to-single-cache", "single-label-to-sequence-label");

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case internalCacheConversion:
		invokeInternalCacheConversion();
		break;
	case externalToCache:
		invokeExternalToCache();
		break;
	case cacheToExternal:
		invokeCacheToExternal();
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

void Application::invokeInternalCacheConversion() {
	switch (Core::Configuration::config(paramInternalCacheConversions_)) {
	case singleLabelToSequenceLabel:
		{
		SingleLabelToSequenceLabelConverter labelConverter;
		labelConverter.convert();
		}
		break;
	case matrixToSingleCache:
		{
		MatrixToSingleCacheConverter matrixConverter;
		matrixConverter.convert();
		}
		break;
	default:
		;
	}
}

void Application::invokeExternalToCache() {

	switch (Core::Configuration::config(paramExternalType_)) {
	case asciiFeatureCache:
		{
		AsciiCacheToBinaryCache asciiConverter;
		asciiConverter.convert();
		}
		break;
	case asciiLabels:
		{
		AsciiLabelConverter labelConverter;
		labelConverter.writeLabelCache();
		}
		break;
	case denseTrajectories:
		{
		DenseTrajectoriesConversion dtConv;
		dtConv.convert();
		}
		break;
	case libSvm:
		std::cerr << "libSvm to cache not implemented. Abort." << std::endl;
		exit(1);
		break;
	default:
		;
	}
}

void Application::invokeCacheToExternal() {

	switch (Core::Configuration::config(paramExternalType_)) {
	case asciiLabels:
		std::cerr << "cache to ascii-labels not implemented. Abort." << std::endl;
		exit(1);
		break;
	case denseTrajectories:
		std::cerr << "cache to dense trajectories not implemented. Abort." << std::endl;
		exit(1);
		break;
	case libSvm:
		{
		LibSvmConverter libSvmConverter;
		libSvmConverter.convert();
		}
		break;
	default:
		;
	}
}
