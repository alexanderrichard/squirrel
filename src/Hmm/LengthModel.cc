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
 * LengthModel.cc
 *
 *  Created on: May 17, 2017
 *      Author: richard
 */

#include "LengthModel.hh"

using namespace Hmm;

const Core::ParameterEnum LengthModel::paramLengthModelType_("type",
		"none, poisson, threshold, linear-decay, monotone-gaussian", "none", "length-model");

const Core::ParameterBool LengthModel::paramIsFramewise_("is-framewise", true, "length-model");

LengthModel::LengthModel() :
		type_((LengthModelType)Core::Configuration::config(paramLengthModelType_)),
		isFramewise_(Core::Configuration::config(paramIsFramewise_))
{}

LengthModel* LengthModel::create() {
	switch ((LengthModelType) Core::Configuration::config(paramLengthModelType_)) {
	case none:
		return new LengthModel();
		break;
	case poisson:
		return new PoissonLengthModel();
		break;
	case threshold:
		return new ThresholdLengthModel();
		break;
	case linearDecay:
		return new LinearDecayLengthModel();
		break;
	case monotoneGaussian:
		return new MonotoneGaussianLengthModel();
		break;
	default:
		return 0; // this can not happen
	}
}

/*
 * PoissonLengthModel
 */
const Core::ParameterString PoissonLengthModel::paramLengthModelFile_("model-file", "", "length-model");

// if true re-scale probabilities such that maximal probability (at lambda) is 1
const Core::ParameterBool PoissonLengthModel::paramRescale_("re-scale", false, "length-model");

PoissonLengthModel::PoissonLengthModel() :
		Precursor(),
		lengthModelFile_(Core::Configuration::config(paramLengthModelFile_)),
		rescale_(Core::Configuration::config(paramRescale_)),
		isInitialized_(false)
{}

void PoissonLengthModel::initialize() {
	if (lengthModelFile_.empty())
		Core::Error::msg("PoissonLengthModel::initialize(): model-file not specified.") << Core::Error::abort;
	lambda_.read(lengthModelFile_);
	rescalingFactor_.resize(lambda_.size());
	rescalingFactor_.setToZero();
	if (rescale_) {
		for (u32 state = 0; state < lambda_.size(); state++) {
			Float logFak = 0;
			for (u32 k = 2; k <= lambda_.at(state); k++)
				logFak += std::log(k);
			rescalingFactor_.at(state) = round(lambda_.at(state)) * std::log(round(lambda_.at(state))) - round(lambda_.at(state)) - logFak;
		}
	}
	isInitialized_ = true;
}

Float PoissonLengthModel::frameScore(u32 length, u32 state) {
	require(isInitialized_);
	require_lt(state, lambda_.size());
	if (!isFramewise_) return 0.0;

	if (length <= round(lambda_.at(state)))
		return 0.0;
	else
		return std::log(lambda_.at(state)) - std::log(length);
}

Float PoissonLengthModel::segmentScore(u32 length, u32 state) {
	require(isInitialized_);
	require_lt(state, lambda_.size());
	if (isFramewise_) return 0.0;

	Float logFak = 0;
	for (u32 k = 2; k <= length; k++)
		logFak += std::log(k);
	return length * std::log(lambda_.at(state)) - lambda_.at(state) - logFak - rescalingFactor_.at(state);
}

/*
 * ThresholdLengthModel
 */
const Core::ParameterFloat ThresholdLengthModel::paramThreshold_("threshold", 100.0, "length-model");

const Core::ParameterFloat ThresholdLengthModel::paramEpsilon_("epsilon", 0.001, "length-model");

ThresholdLengthModel::ThresholdLengthModel() :
		Precursor(),
		threshold_(Core::Configuration::config(paramThreshold_)),
		epsilon_(Core::Configuration::config(paramEpsilon_))
{}

Float ThresholdLengthModel::frameScore(u32 length, u32 state) {
	if (!isFramewise_) return 0.0;

	if (length <= threshold_)
		return 0.0;
	else
		return std::log(epsilon_);
}

Float ThresholdLengthModel::segmentScore(u32 length, u32 state) {
	if (isFramewise_) return 0.0;

	if (length <= threshold_)
		return 0.0;
	else
		return (length - threshold_) * std::log(epsilon_);
}

/*
 * LinearDecayLengthModel
 */
LinearDecayLengthModel::LinearDecayLengthModel() :
		Precursor()
{}

Float LinearDecayLengthModel::frameScore(u32 length, u32 state) {
	if (!isFramewise_) return 0.0;

	if (length <= threshold_)
		return 0.0;
	else if (length <= 2 * threshold_)
		return std::max( (Float)std::log(2 * threshold_ - length) - (Float)std::log(2 * threshold_ - length + 1), epsilon_ );
	else
		return std::log(epsilon_);
}

Float LinearDecayLengthModel::segmentScore(u32 length, u32 state) {
	if (isFramewise_) return 0.0;

	if (length <= threshold_)
		return 0.0;
	else if (length <= 2 * threshold_)
		return std::max( (Float)std::log(1.0 - (length - threshold_) / threshold_), epsilon_ );
	else
		return std::log(epsilon_);
}

/*
 * MonotoneGaussianLengthModel
 */
const Core::ParameterFloat MonotoneGaussianLengthModel::paramMeanLength_("mean-length", 100.0, "length-model");

MonotoneGaussianLengthModel::MonotoneGaussianLengthModel() :
		Precursor(),
		meanLength_(Core::Configuration::config(paramMeanLength_))
{}

Float MonotoneGaussianLengthModel::frameScore(u32 length, u32 state) {
	if (!isFramewise_) return 0.0;
	return -0.5 * (2 * length - 1) / (meanLength_ * meanLength_);
}

Float MonotoneGaussianLengthModel::segmentScore(u32 length, u32 state) {
	if (isFramewise_) return 0.0;
	return -0.5 * (length * length) / (meanLength_ * meanLength_);
}
