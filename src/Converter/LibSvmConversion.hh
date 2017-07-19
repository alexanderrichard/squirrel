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
 * libSvmConversion.hh
 *
 *  Created on: Apr 22, 2014
 *      Author: richard
 */

#ifndef CONVERTER_LIBSVMCONVERSION_HH_
#define CONVERTER_LIBSVMCONVERSION_HH_

#include "Core/CommonHeaders.hh"
#include "Features/AlignedFeatureReader.hh"

namespace Converter {

class LibSvmConverter
{
private:
	static const Core::ParameterString paramLibSvmFile_;
	static const Core::ParameterBool paramUsePrecomputedKernelFormat_;
	std::string libSvmFile_;
	bool usePrecomputedKernelFormat_;
	Features::LabeledFeatureReader featureReader_;
public:
	LibSvmConverter();
	void convert();
};

} // namespace

#endif /* CONVERTER_LIBSVMCONVERSION_HH_ */
