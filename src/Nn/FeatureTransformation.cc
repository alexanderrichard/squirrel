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
 * Preprocessor.cc
 *
 *  Created on: May 23, 2017
 *      Author: richard
 */

#include "FeatureTransformation.hh"

using namespace Nn;

const Core::ParameterEnum FeatureTransformation::paramTransformationType_("type", "none, vector-to-sequence, sequence-to-vector", "none", "neural-network.feature-transformation");

const Core::ParameterInt FeatureTransformation::paramTransformedFeatureDimension_("transformed-feature-dimension", 0, "neural-network.feature-transformation");

FeatureTransformation::FeatureTransformation(FeatureType originalType) :
		type_((TransformationType) Core::Configuration::config(paramTransformationType_)),
		dimension_(Core::Configuration::config(paramTransformedFeatureDimension_)),
		originalType_(originalType)
{}

void FeatureTransformation::transform(Matrix& in, MatrixContainer& out) {
	if (dimension_ == 0)
		Core::Error::msg("Nn::FeatureTransformation: neural-network.transformed-feature-dimension must not be 0.") << Core::Error::abort;

	require_eq(type_, vectorToSequence);
	require_eq(in.nRows() % dimension_, 0);

	in.initComputation();
	out.initComputation(false);
	out.reset();
	u32 nTimeframes = in.nRows() / dimension_;
	out.setMaximalMemory(nTimeframes);
	for (u32 t = 0; t < nTimeframes; t++) {
		out.addTimeframe(dimension_, in.nColumns());
		out.getLast().copyBlockFromMatrix(in, t * dimension_, 0, 0, 0, dimension_, in.nColumns());
	}
	in.finishComputation(false);
	out.finishComputation();
}

void FeatureTransformation::transform(MatrixContainer& in, Matrix& out) {
	if (dimension_ == 0)
		Core::Error::msg("Nn::FeatureTransformation: neural-network.transformed-feature-dimension must not be 0.") << Core::Error::abort;

	require_eq(type_, sequenceToVector);
	require_eq(in.getLast().nRows() * in.nTimeframes(), dimension_);
	in.initComputation();
	out.initComputation(false);
	out.resize(dimension_, in.getLast().nColumns());
	out.setToZero();
	for (u32 t = 0; t < in.nTimeframes(); t++) {
		out.copyBlockFromMatrix(in.at(t), 0, 0, t * in.at(t).nRows(), 0, in.at(t).nRows(), in.at(t).nColumns());
	}
	in.finishComputation(false);
	out.finishComputation();
}

FeatureType FeatureTransformation::outputFormat() const {
	if (type_ == vectorToSequence)
		return sequence;
	else if (type_ == sequenceToVector)
		return single;
	else // type_ == none
		return originalType_;
}
