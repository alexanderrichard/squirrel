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
 * KMeans.hh
 *
 *  Created on: Apr 9, 2014
 *      Author: richard
 */

#ifndef CLUSTERING_KMEANS_HH_
#define CLUSTERING_KMEANS_HH_

#include "Features/FeatureReader.hh"
#include "Core/CommonHeaders.hh"
#include "ClusteringAlgorithm.hh"
#include "Math/CudaVector.hh"
#include "Math/CudaMatrix.hh"
#include <string.h>
#include <vector>

namespace Clustering {


/**
 * KMeans clustering algorithm
 */

class KMeans : public ClusteringAlgorithm
{
private:
	static const Core::ParameterInt paramNumberOfClusters_;
	static const Core::ParameterString paramMeanInputFile_;
	static const Core::ParameterString paramMeanOutputFile_;
	static const Core::ParameterInt paramMaxIterations_;
	static const Core::ParameterFloat paramStopThreshold_;
	static const Core::ParameterInt paramBatchSize_;
	static const Core::ParameterEnum paramInitializationMode_;
	enum InitializationMode { random };
private:
	u32 nClusters_;
	std::string meanInputFile_;
	std::string meanOutputFile_;
	u32 maxIterations_;
	Float stopThreshold_;
	u32 batchSize_;

	Features::FeatureReader featureReader_;

	Math::CudaMatrix<Float> data_;
	Math::CudaMatrix<Float> means_;
	Math::CudaMatrix<Float> dist_;
	Math::CudaVector<Float> tmp_;
	Math::CudaVector<u32> batchClusterAssignment_;
	Math::CudaVector<Float> clusterCount_;
	std::vector<u32> clusterAssignment_;

	InitializationMode initializationMode_;

	u32 featureDim_;
	u32 nObservations_;

	bool initializedMeans_;

	void bufferFeature(const Math::Vector<Float>& f, u32 index);

	void randomInitialization();
	void initializeSeeds();
	void distanceAndClusterAssignment();
	void writeParameters();
	void generateBatch();
	Float clusterBatch(u32 startIndex, u32 endIndex);
public:
	KMeans();
	virtual ~KMeans() {}

	// initialize needs to be called explicitly if no clustering is generated, but getClusterIndex is used
	virtual void initialize();
	virtual u32 nClusters();
	virtual void generateClustering();
};

} // namespace

#endif /* CLUSTERING_KMEANS_HH_ */
