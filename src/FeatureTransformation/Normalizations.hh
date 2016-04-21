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

#ifndef FEATURETRANSFORMATION_NORMALIZATIONS_HH_
#define FEATURETRANSFORMATION_NORMALIZATIONS_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Features/FeatureReader.hh>
#include <Features/FeatureWriter.hh>

namespace FeatureTransformation {

/*
 * estimate the mean and standard deviation of the given input features
 */
class MeanAndVarianceEstimation {
private:
	static const Core::ParameterString paramMeanFile_;
	static const Core::ParameterString paramStandardDeviationFile_;
	static const Core::ParameterFloat paramMinStandardDeviation_;
private:
	std::string meanFile_;
	std::string standardDeviationFile_;
	Math::Vector<Float> mean_;
	Math::Vector<Float> standardDeviation_;
	Float minStandardDeviation_;
public:
	MeanAndVarianceEstimation();
	~MeanAndVarianceEstimation() {}
	void estimate();
};

/*
 * compute min and max value of each feature in the given input feature cache
 */
class MinMaxEstimation {
private:
	static const Core::ParameterString paramMinFile_;
	static const Core::ParameterString paramMaxFile_;
private:
	std::string minFile_;
	std::string maxFile_;
	Math::Vector<Float> min_;
	Math::Vector<Float> max_;
public:
	MinMaxEstimation();
	~MinMaxEstimation() {}
	void estimate();
};

} // namespace

#endif /* FEATURETRANSFORMATION_NORMALIZATIONS_HH_ */
