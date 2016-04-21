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
