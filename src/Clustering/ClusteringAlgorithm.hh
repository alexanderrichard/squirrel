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
 * ClusteringAlgorithm.hh
 *
 *  Created on: Apr 9, 2014
 *      Author: richard
 */

#ifndef CLUSTERING_CLUSTERINGALGORITHM_HH_
#define CLUSTERING_CLUSTERINGALGORITHM_HH_

#include "Math/Vector.hh"
#include "Features/FeatureReader.hh"

namespace Clustering {

class ClusteringAlgorithm
{
public:
	typedef u32 ClusterIndex;

	virtual ~ClusteringAlgorithm() {}

	/*
	 * initialize the clustering algorithm
	 */
	virtual void initialize() = 0;
	/*
	 * @return the number of clusters
	 */
	virtual u32 nClusters() = 0;
	/*
	 * generate clusters based on the features in the feature cache associated with featureReader
	 */
	virtual void generateClustering() = 0;
};

} // namespace

#endif /* CLUSTERING_CLUSTERINGALGORITHM_HH_ */
