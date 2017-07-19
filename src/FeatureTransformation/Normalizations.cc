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
 * StandardScoreNormalization.cc
 *
 *  Created on: Jun 3, 2014
 *      Author: richard
 */

#include "Normalizations.hh"
#include <cmath>

using namespace FeatureTransformation;

/*
 * MeanAndVarianceEstimation
 */
const Core::ParameterString MeanAndVarianceEstimation::paramMeanFile_("mean-file", "",
		"feature-transformation.mean-and-variance-estimation");

const Core::ParameterString MeanAndVarianceEstimation::paramStandardDeviationFile_("standard-deviation-file", "",
		"feature-transformation.mean-and-variance-estimation");

const Core::ParameterFloat MeanAndVarianceEstimation::paramMinStandardDeviation_("min-standard-deviation", 0.000001,
		"feature-transformation.mean-and-variance-estimation");

MeanAndVarianceEstimation::MeanAndVarianceEstimation() :
		meanFile_(Core::Configuration::config(paramMeanFile_)),
		standardDeviationFile_(Core::Configuration::config(paramStandardDeviationFile_)),
		minStandardDeviation_(Core::Configuration::config(paramMinStandardDeviation_))
{}

void MeanAndVarianceEstimation::estimate() {
	require(!meanFile_.empty());
	require(!standardDeviationFile_.empty());
	Core::Log::openTag("standard-score-normalization");

	Features::FeatureReader featureReader;
	featureReader.initialize();
	mean_.resize(featureReader.featureDimension());
	standardDeviation_.resize(featureReader.featureDimension());
	mean_.setToZero();
	standardDeviation_.setToZero();

	// run over the data set in a two-pass fashion for numerical stability
	while (featureReader.hasFeatures()) {
		mean_.add(featureReader.next());
	}
	mean_.scale(1.0 / featureReader.totalNumberOfFeatures());

	featureReader.newEpoch();
	Math::Vector<Float> tmp(featureReader.featureDimension());
	while (featureReader.hasFeatures()) {
		tmp.copy(featureReader.next());
		tmp.add(mean_, (Float)-1.0);
		tmp.elementwiseMultiplication(tmp);
		standardDeviation_.add(tmp);
	}
	standardDeviation_.scale(1.0 / featureReader.totalNumberOfFeatures());
	// note that here we still have the variance
	standardDeviation_.ensureMinimalValue(minStandardDeviation_ * minStandardDeviation_);
	// compute standard deviation
	for (u32 i = 0; i < standardDeviation_.nRows(); i++) {
		standardDeviation_.at(i) = sqrt(standardDeviation_.at(i));
	}

	Core::Log::os("write mean to ") << meanFile_;
	mean_.write(meanFile_);
	Core::Log::os("write standard deviation to ") << standardDeviationFile_;
	standardDeviation_.write(standardDeviationFile_);

	Core::Log::closeTag();
}

/*
 * MinMaxEstimation
 */
const Core::ParameterString MinMaxEstimation::paramMinFile_("min-file", "",
		"feature-transformation.min-max-estimation");

const Core::ParameterString MinMaxEstimation::paramMaxFile_("max-file", "",
		"feature-transformation.min-max-estimation");

MinMaxEstimation::MinMaxEstimation() :
		minFile_(Core::Configuration::config(paramMinFile_)),
		maxFile_(Core::Configuration::config(paramMaxFile_))
{}

void MinMaxEstimation::estimate() {
	require(!minFile_.empty());
	require(!maxFile_.empty());
	Core::Log::openTag("min-max-estimation");

	Features::FeatureReader featureReader;
	featureReader.initialize();
	min_.resize(featureReader.featureDimension());
	max_.resize(featureReader.featureDimension());
	min_.fill(Types::max<Float>());
	max_.fill(Types::min<Float>());

	while (featureReader.hasFeatures()) {
		const Math::Vector<Float>& f = featureReader.next();
		for (u32 d = 0; d < featureReader.featureDimension(); d++) {
			min_.at(d) = (f.at(d) < min_.at(d) ? f.at(d) : min_.at(d));
			max_.at(d) = (f.at(d) > max_.at(d) ? f.at(d) : max_.at(d));
		}
	}

	Core::Log::os("write min to ") << minFile_;
	min_.write(minFile_);
	Core::Log::os("write max to ") << maxFile_;
	max_.write(maxFile_);

	Core::Log::closeTag();
}
