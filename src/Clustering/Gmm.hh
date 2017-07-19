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
 * Gmm.hh
 *
 *  Created on: Jan 13, 2015
 *      Author: richard
 */

#ifndef CLUSTERING_GMM_HH_
#define CLUSTERING_GMM_HH_

#include <Core/CommonHeaders.hh>
#include "ClusteringAlgorithm.hh"
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Features/FeatureReader.hh>

namespace Clustering {

class GmmBase
{
private:
	static const Core::ParameterString paramMeanInputFile_;
	static const Core::ParameterString paramVarianceInputFile_;
	static const Core::ParameterString paramWeightsInputFile_;
	static const Core::ParameterString paramMeanOutputFile_;
	static const Core::ParameterString paramVarianceOutputFile_;
	static const Core::ParameterString paramWeightsOutputFile_;
protected:
	std::string meanInputFile_;
	std::string varianceInputFile_;
	std::string weightsInputFile_;
	std::string meanOutputFile_;
	std::string varianceOutputFile_;
	std::string weightsOutputFile_;
public:
	GmmBase();
	virtual ~GmmBase() {}
};

class GmmTrainer : public GmmBase, public ClusteringAlgorithm
{
private:
	static const Core::ParameterFloat paramMinVariance_;
	static const Core::ParameterInt paramMaxIterations_;
	static const Core::ParameterInt paramNumberOfDensities_;
	static const Core::ParameterInt paramBatchSize_;
	static const Core::ParameterBool paramMaximumApproximation_;
private:
	Math::CudaMatrix<Float> oldMu_;
	Math::CudaMatrix<Float> oldSigma_;
	Math::CudaVector<Float> oldWeights_;
	Math::CudaMatrix<Float> newMu_;
	Math::CudaMatrix<Float> newSigma_;
	Math::CudaVector<Float> newWeights_;

	Features::FeatureReader featureReader_;

	// additional containers
	Math::CudaMatrix<Float> posteriors_;
	Math::CudaMatrix<Float> buffer_;
	Math::CudaMatrix<Float> lambda_;
	Math::CudaVector<Float> bias_;
	Math::CudaMatrix<Float> tmpMatrix_;
	Math::CudaVector<Float> tmpVector_;

	Float minVariance_;
	u32 maxIterations_;
	u32 nMixtures_;
	u32 batchSize_;
	bool maximumApproximation_;
	Float logLik_;
private:
	void writeParameters();
	void initializeParameters();
	void transformForSoftmax();
	void computePosteriors();
	void updateParameters();
	void finalizeEstimation();
	void bufferFeatures();
public:
	GmmTrainer();
	virtual ~GmmTrainer() {}

	virtual void initialize();
	virtual u32 nClusters() { return nMixtures_; }
	virtual void generateClustering();
};

class GmmDensitySplitter : public GmmBase
{
private:
	Math::Matrix<Float> oldMu_;
	Math::Matrix<Float> oldSigma_;
	Math::Vector<Float> oldWeights_;
	Math::Matrix<Float> newMu_;
	Math::Matrix<Float> newSigma_;
	Math::Vector<Float> newWeights_;

	void readParameters();
	void writeParameters();
public:
	GmmDensitySplitter() {}
	virtual ~GmmDensitySplitter() {}
	void split();
};

} // namespace

#endif /* GMM_HH_ */
