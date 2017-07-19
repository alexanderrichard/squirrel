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
 * Preprocessor.hh
 *
 *  Created on: May 23, 2017
 *      Author: richard
 */

#ifndef NN_PREPROCESSOR_HH_
#define NN_PREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "MatrixContainer.hh"

namespace Nn {

class FeatureTransformation
{
private:
	static const Core::ParameterEnum paramTransformationType_;
	static const Core::ParameterInt paramTransformedFeatureDimension_;
public:
	enum TransformationType { none, vectorToSequence, sequenceToVector };
protected:
	TransformationType type_;
	u32 dimension_;
	FeatureType originalType_;
public:
	FeatureTransformation(FeatureType originalType);
	virtual ~FeatureTransformation() {}

	void transform(Matrix& in, MatrixContainer& out);
	void transform(MatrixContainer& in, Matrix& out);

	FeatureType outputFormat() const;
};

} // namespace

#endif /* NN_PREPROCESSOR_HH_ */
