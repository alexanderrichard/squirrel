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


#include "LogLinearConverter.hh"
#include <Features/FeatureReader.hh>
#include <math.h>
#include <fstream>

using namespace Converter;

/*
 * GaussianMixtureToLogLinear
 */
const Core::ParameterString GaussianMixtureToLogLinear::paramMeanInputFile_("mean-input-file", "", "converter");
const Core::ParameterString GaussianMixtureToLogLinear::paramCovarianceInputFile_("variance-input-file", "", "converter");
const Core::ParameterString GaussianMixtureToLogLinear::paramWeightInputFile_("weight-input-file", "", "converter");
const Core::ParameterString GaussianMixtureToLogLinear::paramAlphaOutputFile_("log-linear-bias-output-file", "", "converter");
const Core::ParameterString GaussianMixtureToLogLinear::paramLambdaOutputFile_("log-linear-weights-output-file", "", "converter");

GaussianMixtureToLogLinear::GaussianMixtureToLogLinear() :
		meanInputFile_(Core::Configuration::config(paramMeanInputFile_)),
		covarianceInputFile_(Core::Configuration::config(paramCovarianceInputFile_)),
		weightInputFile_(Core::Configuration::config(paramWeightInputFile_)),
		alphaOutputFile_(Core::Configuration::config(paramAlphaOutputFile_)),
		lambdaOutputFile_(Core::Configuration::config(paramLambdaOutputFile_))
{}

void GaussianMixtureToLogLinear::convert() {
	Math::Matrix<Float> means;
	Math::Matrix<Float> covariances;
	Math::Vector<Float> weights;

	require(!meanInputFile_.empty());
	means.read(meanInputFile_);
	require(!covarianceInputFile_.empty());
	covariances.read(covarianceInputFile_);
	require(!weightInputFile_.empty());
	weights.read(weightInputFile_);

	u32 nClusters = means.nRows();
	u32 featureDim = means.nColumns();

	Math::Matrix<Float> lambda(2*featureDim, nClusters);
	Math::Vector<Float> alpha(nClusters);

	alpha.setToZero();

	for (u32 c = 0; c < nClusters; c++) {
		for (u32 f = 0; f < featureDim; f++)
		{
			lambda.at(f, c) = means.at(c, f) / covariances.at(c, f);
			lambda.at(f + featureDim, c) = -0.5 / covariances.at(c,f);
			alpha.at(c) +=  pow(means.at(c,f), 2) * 1.0/ covariances.at(c, f) + log(covariances.at(c, f));
		}
		alpha.at(c) = log(weights.at(c)) -0.5 * (alpha.at(c) + featureDim * log(2*M_PI));
	}

	require(!alphaOutputFile_.empty());
	alpha.write(alphaOutputFile_);
	require(!lambdaOutputFile_.empty());
	lambda.write(lambdaOutputFile_);
}

/*
 * KMeansToLogLinear
 */
const Core::ParameterString KMeansToLogLinear::paramMeanInputFile_("mean-input-file", "", "converter");
const Core::ParameterString KMeansToLogLinear::paramAlphaOutputFile_("log-linear-bias-output-file", "", "converter");
const Core::ParameterString KMeansToLogLinear::paramLambdaOutputFile_("log-linear-weights-output-file", "", "converter");

KMeansToLogLinear::KMeansToLogLinear() :
		meanInputFile_(Core::Configuration::config(paramMeanInputFile_)),
		alphaOutputFile_(Core::Configuration::config(paramAlphaOutputFile_)),
		lambdaOutputFile_(Core::Configuration::config(paramLambdaOutputFile_))
{}

void KMeansToLogLinear::convert() {
	Math::Matrix<Float> means;

	require(!meanInputFile_.empty());
	means.read(meanInputFile_);

	u32 nClusters = means.nRows();
	u32 featureDim = means.nColumns();

	Math::Matrix<Float> lambda(featureDim, nClusters);
	Math::Vector<Float> alpha(nClusters);

	alpha.setToZero();

	for (u32 c = 0; c < nClusters; c++) {
		for (u32 f = 0; f < featureDim; f++)
		{
			lambda.at(f, c) = means.at(c, f);
			alpha.at(c) -=  0.5 * means.at(c,f) * means.at(c,f);
		}
	}

	require(!alphaOutputFile_.empty());
	alpha.write(alphaOutputFile_);
	require(!lambdaOutputFile_.empty());
	lambda.write(lambdaOutputFile_);
}

/*
 * LibSvmToLogLinear
 */
const Core::ParameterString LibSvmToLogLinear::paramRhoFile_("rho-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramSupportVectorCoefficientsFile_("support-vector-coefficients-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramSupportVectorIndexFile_("support-vector-index-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramProbAFile_("probA-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramProbBFile_("probB-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramBiasFile_("bias-file", "", "converter");

const Core::ParameterString LibSvmToLogLinear::paramWeightsFile_("weights-file", "", "converter");

LibSvmToLogLinear::LibSvmToLogLinear() :
		rhoFile_(Core::Configuration::config(paramRhoFile_)),
		supportVectorCoefficientsFile_(Core::Configuration::config(paramSupportVectorCoefficientsFile_)),
		supportVectorIndexFile_(Core::Configuration::config(paramSupportVectorIndexFile_)),
		probAFile_(Core::Configuration::config(paramProbAFile_)),
		probBFile_(Core::Configuration::config(paramProbBFile_)),
		biasFile_(Core::Configuration::config(paramBiasFile_)),
		weightsFile_(Core::Configuration::config(paramWeightsFile_))
{}

void LibSvmToLogLinear::readData() {
	// read the training data
	Features::FeatureReader fr;
	fr.initialize();
	trainingData_.resize(fr.featureDimension(), fr.totalNumberOfFeatures());
	u32 idx = 0;
	while (fr.hasFeatures()) {
		const Math::Vector<Float>& f = fr.next();
		for (u32 d = 0; d < f.size(); d++)
			trainingData_.at(d, idx) = f.at(d);
		idx++;
	}
	// read probA and probB
	require(!probAFile_.empty());
	probA_.read(probAFile_);
	require(!probBFile_.empty());
	probB_.read(probBFile_);
	// read the bias
	require(!rhoFile_.empty());
	bias_.read(rhoFile_);
	bias_.scale(-1.0);
	require_eq(bias_.size(), probA_.size());
	require_eq(bias_.size(), probB_.size());
	// read the support vector coefficients
	require(!supportVectorCoefficientsFile_.empty());
	svCoef_.resize(bias_.size());
	std::ifstream f_svCoef(supportVectorCoefficientsFile_.c_str());
	std::string line;
	u32 c = 0;
	while (std::getline(f_svCoef, line)) {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, line);
		svCoef_.at(c).resize(tmp.size());
		for (u32 i = 0; i < tmp.size(); i++) {
			svCoef_.at(c).at(i) = atof(tmp.at(i).c_str());
		}
		c++;
	}
	f_svCoef.close();
	// read the support vector indices
	require(!supportVectorIndexFile_.empty());
	svIdx_.resize(bias_.size());
	std::ifstream f_svIdx(supportVectorIndexFile_.c_str());
	c = 0;
	while (std::getline(f_svIdx, line)) {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, line);
		svIdx_.at(c).resize(tmp.size());
		for (u32 i = 0; i < tmp.size(); i++) {
			svIdx_.at(c).at(i) = atoi(tmp.at(i).c_str());
		}
		c++;
	}
	f_svIdx.close();
	// ensure consistency of the files
	for (u32 i = 0; i < bias_.size(); i++) {
		require_eq(svCoef_.at(i).size(), svIdx_.at(i).size());
	}
}

void LibSvmToLogLinear::convert() {
	readData();
	weights_.resize(trainingData_.nRows(), bias_.size());
	weights_.setToZero();
	for (u32 c = 0; c < bias_.size(); c++) {
		// accumulate all support vectors for this class
		for (u32 sv = 0; sv < svIdx_.at(c).size(); sv++) {
#pragma omp parallel for
			for (u32 d = 0; d < weights_.nRows(); d++) {
				weights_.at(d, c) += trainingData_.at(d, svIdx_.at(c).at(sv)) * svCoef_.at(c).at(sv);
			}
		}
	}
	// prepare for use with sigmoid function
	weights_.multiplyColumnsByScalars(probA_);
	bias_.elementwiseMultiplication(probA_);
	bias_.add(probB_);
	// save result
	require(!biasFile_.empty());
	require(!weightsFile_.empty());
	bias_.write(biasFile_);
	weights_.write(weightsFile_);
}


