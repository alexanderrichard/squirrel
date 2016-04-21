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

#ifndef CONVERTER_LIBSVMCONVERSION_HH_
#define CONVERTER_LIBSVMCONVERSION_HH_

#include "Core/CommonHeaders.hh"
#include "Features/LabeledFeatureReader.hh"

namespace Converter {

/*
 * convert a feature cache to a file format libSVM can read
 */
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

/*
 * convert a libSVM model to a weight matrix and bias vector that can be used as parameters
 * for a log-linear model (i.e. a neural network without hidden layer)
 */
class LibSvmToLogLinear
{
private:
	static const Core::ParameterString paramRhoFile_;
	static const Core::ParameterString paramSupportVectorCoefficientsFile_;
	static const Core::ParameterString paramSupportVectorIndexFile_;
	static const Core::ParameterString paramProbAFile_;
	static const Core::ParameterString paramProbBFile_;
	static const Core::ParameterString paramBiasFile_;
	static const Core::ParameterString paramWeightsFile_;
	std::string rhoFile_;
	std::string supportVectorCoefficientsFile_;
	std::string supportVectorIndexFile_;
	std::string probAFile_;
	std::string probBFile_;
	std::string biasFile_;
	std::string weightsFile_;
	Math::Matrix<Float> trainingData_;
	std::vector< Math::Vector<Float> > svCoef_;
	std::vector< Math::Vector<u32> > svIdx_;
	Math::Vector<Float> probA_;
	Math::Vector<Float> probB_;
	Math::Vector<Float> bias_;
	Math::Matrix<Float> weights_;
	void readData();
public:
	LibSvmToLogLinear();
	void convert();
};

} // namespace

#endif /* CONVERTER_LIBSVMCONVERSION_HH_ */
