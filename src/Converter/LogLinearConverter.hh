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


#ifndef LOGLINEARCONVERTER_HH_
#define LOGLINEARCONVERTER_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Matrix.hh>
#include <Math/Vector.hh>

namespace Converter {

class GaussianMixtureToLogLinear
{
private:
	static const Core::ParameterString paramMeanInputFile_;
	static const Core::ParameterString paramCovarianceInputFile_;
	static const Core::ParameterString paramWeightInputFile_;
	static const Core::ParameterString paramAlphaOutputFile_;
	static const Core::ParameterString paramLambdaOutputFile_;
	std::string meanInputFile_;
	std::string covarianceInputFile_;
	std::string weightInputFile_;
	std::string alphaOutputFile_;
	std::string lambdaOutputFile_;
public:
	GaussianMixtureToLogLinear();
	void convert();
};

/*
 * special case of gaussian mixture: kMeans model
 */
class KMeansToLogLinear
{
private:
	static const Core::ParameterString paramMeanInputFile_;
	static const Core::ParameterString paramAlphaOutputFile_;
	static const Core::ParameterString paramLambdaOutputFile_;
	std::string meanInputFile_;
	std::string alphaOutputFile_;
	std::string lambdaOutputFile_;
public:
	KMeansToLogLinear();
	void convert();
};

/*
 * LibSvmToLogLinear
 * assumes sigmoid instead of softmax
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



#endif /* LOGLINEARCONVERTER_HH_ */
