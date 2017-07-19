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
 * MultiChannelRbfChiSquareKernel.hh
 *
 *  Created on: Apr 16, 2014
 *      Author: richard
 */

#ifndef FEATURETRANSFORMATION_MULTICHANNELRBFCHISQUAREKERNEL_HH_
#define FEATURETRANSFORMATION_MULTICHANNELRBFCHISQUAREKERNEL_HH_

#include "Core/CommonHeaders.hh"
#include "Math/Vector.hh"
#include "Kernel.hh"
#include <vector>

namespace FeatureTransformation {

class MultiChannelRbfChiSquareKernel : public Kernel
{
private:
	/*
	 * provide number of channels in config
	 * for each channel, the feature cache must be of type "single" and can be defined via
	 * feature-transformation.multi-channel-rbf-chi-square-kernel.channel-<n>-train.feature-cache=<filename> and
	 * feature-transformation.multi-channel-rbf-chi-square-kernel.channe-<n>-test.feature-cache=<filename>
	 * (to process the training data, train-cache-file and test-cache-file may be the same cache)
	 * each cache must have the same amount of observations
	 */
	static const Core::ParameterInt paramNumberOfChannels_;
	static const Core::ParameterString paramMeanDistancesFile_;
	static const Core::ParameterBool paramEstimateKernelParameters_;

	typedef Kernel Precursor;

	u32 nChannels_;
	std::vector<Features::FeatureReader*> featureReaderTrain_;
	std::vector<Features::FeatureReader*> featureReaderTest_;
	Math::Vector<Float> meanDistances_;

	std::string meanDistancesFile_;
	bool estimateKernelParameters_;

	void estimateKernelParameters();
	void _applyKernel();
public:
	MultiChannelRbfChiSquareKernel();
	virtual ~MultiChannelRbfChiSquareKernel();

	virtual void initialize();
	virtual void finalize();

	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output);

	/*
	 * apply kernel to all features x_i (K(x_i,x_j) for all x_i in test cache and all x_j in train cache) and write resulting cache
	 */
	virtual void applyKernel();
};

} // namespace


#endif /* FEATURETRANSFORMATION_MULTICHANNELRBFCHISQUAREKERNEL_HH_ */
