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
 * MultiChannelRbfChiSquareKernel.cc
 *
 *  Created on: Apr 16, 2014
 *      Author: richard
 */

#include "MultiChannelRbfChiSquareKernel.hh"
#include <sstream>
#include <math.h>

using namespace FeatureTransformation;

const Core::ParameterInt MultiChannelRbfChiSquareKernel::paramNumberOfChannels_("number-of-channels", 1,
		"feature-transformation.multi-channel-rbf-chi-square-kernel");

const Core::ParameterString MultiChannelRbfChiSquareKernel::paramMeanDistancesFile_("mean-distances", "",
		"feature-transformation.multi-channel-rbf-chi-square-kernel");

const Core::ParameterBool MultiChannelRbfChiSquareKernel::paramEstimateKernelParameters_("estimate-kernel-parameters", false,
		"feature-transformation.multi-channel-rbf-chi-square-kernel");

MultiChannelRbfChiSquareKernel::MultiChannelRbfChiSquareKernel() :
		nChannels_(Core::Configuration::config(paramNumberOfChannels_)),
		featureReaderTrain_(nChannels_, 0),
		featureReaderTest_(nChannels_, 0),
		meanDistancesFile_(Core::Configuration::config(paramMeanDistancesFile_)),
		estimateKernelParameters_(Core::Configuration::config(paramEstimateKernelParameters_))
{}

MultiChannelRbfChiSquareKernel::~MultiChannelRbfChiSquareKernel() {
	for (u32 i = 0; i < nChannels_; i++) {
		if (featureReaderTrain_.at(i))
			delete featureReaderTrain_.at(i);;
		if (featureReaderTest_.at(i))
			delete featureReaderTest_.at(i);
	}
}

void MultiChannelRbfChiSquareKernel::initialize() {
	if (!isInitialized_) {
		require(featureReaderTrain_.size() == nChannels_);
		require(featureReaderTest_.size() == nChannels_);
		for (u32 i = 0; i < nChannels_; i++) {
			std::stringstream sTrain, sTest;
			sTrain << "feature-transformation.multi-channel-rbf-chi-square-kernel.channel-" << i+1 << "-train";
			sTest << "feature-transformation.multi-channel-rbf-chi-square-kernel.channel-" << i+1 << "-test";
			featureReaderTrain_.at(i) = new Features::FeatureReader(sTrain.str().c_str());
			featureReaderTest_.at(i) = new Features::FeatureReader(sTest.str().c_str());
			featureReaderTrain_.at(i)->initialize();
			featureReaderTest_.at(i)->initialize();
		}
		// read mean distances file, if not in parameter estimation mode
		if (!estimateKernelParameters_) {
			featureWriter_.initialize(featureReaderTest_.at(0)->totalNumberOfFeatures(), featureReaderTrain_.at(0)->totalNumberOfFeatures());
			meanDistances_.read(meanDistancesFile_);
			require(meanDistances_.nRows() == nChannels_);
		}
		else {
			meanDistances_.resize(nChannels_);
			meanDistances_.setToZero();
		}
		isInitialized_ = true;
	}
}

void MultiChannelRbfChiSquareKernel::finalize() {
	if (!isFinalized_) {
		for (u32 i = 0; i < nChannels_; i++) {
			if (featureReaderTrain_.at(i))
				delete featureReaderTrain_.at(i);
			if (featureReaderTest_.at(i))
				delete featureReaderTest_.at(i);
			featureReaderTrain_.at(i) = 0;
			featureReaderTest_.at(i) = 0;
		}
		if (!estimateKernelParameters_)
			featureWriter_.finalize();
		isFinalized_ = true;
	}
}

void MultiChannelRbfChiSquareKernel::estimateKernelParameters() {
	require(isInitialized_);
	for (u32 i = 0; i < nChannels_; i++) {
		u32 nDistances = 0;
		while (featureReaderTest_.at(i)->hasFeatures()) {
			const Math::Vector<Float>& f = featureReaderTest_.at(i)->next();
			while (featureReaderTrain_.at(i)->hasFeatures()) {
				meanDistances_.at(i) += f.chiSquareDistance(featureReaderTrain_.at(i)->next());
				nDistances++;
			}
			featureReaderTrain_.at(i)->newEpoch();
		}
		meanDistances_.at(i) /= nDistances;
	}
	meanDistances_.write(meanDistancesFile_);
}

void MultiChannelRbfChiSquareKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) {

	std::cerr << "MultiChannelRbfChiSquareKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output):" << std::endl;
	std::cerr << "This method is not available for multi-channel kernels. Abort." << std::endl;
	exit(1);

}

void MultiChannelRbfChiSquareKernel::_applyKernel() {

	// require that all channels have the same amount of observations
	for (u32 i = 1; i < nChannels_; i++) {
		require(featureReaderTrain_.at(0)->totalNumberOfFeatures() == featureReaderTrain_.at(i)->totalNumberOfFeatures());
		require(featureReaderTest_.at(0)->totalNumberOfFeatures() == featureReaderTest_.at(i)->totalNumberOfFeatures());
	}

	// process each test observation (x_i)
	while (featureReaderTest_.at(0)->hasFeatures()) {

		// compute K(x_i, x_j) for all training examples x_j
		Math::Vector<Float> result(featureReaderTrain_.at(0)->totalNumberOfFeatures());
		result.setToZero();
		for (u32 channel = 0; channel < nChannels_; channel++) {
			const Math::Vector<Float>& f = featureReaderTest_.at(channel)->next();
			u32 index = 0;
			featureReaderTrain_.at(channel)->newEpoch();
			while (featureReaderTrain_.at(channel)->hasFeatures()) {
				result.at(index) += f.chiSquareDistance(featureReaderTrain_.at(channel)->next()) / meanDistances_.at(channel);
				index++;
			}
		}

		// compute exponential
		result.scale(-1.0 / (Float)nChannels_);
		for (u32 i = 0; i < result.nRows(); i++) {
			result.at(i) = exp(result.at(i));
		}

		// write result
		featureWriter_.write(result);
	}
}

void MultiChannelRbfChiSquareKernel::applyKernel() {
	require(isInitialized_);

	if (estimateKernelParameters_) {
		estimateKernelParameters();
	}
	else {
		_applyKernel();
	}
}
