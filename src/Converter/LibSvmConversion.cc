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
 * LibSvmConversion.cc
 *
 *  Created on: Apr 22, 2014
 *      Author: richard
 */

#include "LibSvmConversion.hh"
#include "Math/Vector.hh"

using namespace Converter;

const Core::ParameterString LibSvmConverter::paramLibSvmFile_("lib-svm-file", "", "converter");

const Core::ParameterBool LibSvmConverter::paramUsePrecomputedKernelFormat_("use-precomputed-kernel-format", false,
		"converter");

LibSvmConverter::LibSvmConverter() :
		libSvmFile_(Core::Configuration::config(paramLibSvmFile_)),
		usePrecomputedKernelFormat_(Core::Configuration::config(paramUsePrecomputedKernelFormat_))
{}

void LibSvmConverter::convert() {
	Core::AsciiStream libSvmStream(libSvmFile_, std::ios::out);
	featureReader_.initialize();
	u32 index = 0;
	while (featureReader_.hasFeatures()) {
		const Math::Vector<Float>& v = featureReader_.next();
		libSvmStream << featureReader_.label();
		if (usePrecomputedKernelFormat_) {
			libSvmStream << " " << "0:" << index+1;
		}
		for (u32 i = 0; i < featureReader_.featureDimension(); i++) {
			libSvmStream << " " << i+1 << ":" << v.at(i);
		}
		libSvmStream << Core::IOStream::endl;
		index++;
	}
	libSvmStream.close();
}
