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
 * KMeans.cc
 *
 *  Created on: Apr 9, 2014
 *      Author: richard
 */

#include "KMeans.hh"
#include "Math/Random.hh"
#include <algorithm>

using namespace Clustering;

const Core::ParameterInt KMeans::paramNumberOfClusters_("number-of-clusters", 1, "clustering.k-means");

const Core::ParameterString KMeans::paramMeanInputFile_("mean-input-file", "", "clustering.k-means");

const Core::ParameterString KMeans::paramMeanOutputFile_("mean-output-file", "", "clustering.k-means");

const Core::ParameterInt KMeans::paramMaxIterations_("max-number-of-iterations", Types::max<u32>(), "clustering.k-means");

const Core::ParameterFloat KMeans::paramStopThreshold_("stop-if-score-improves-by-less-than", 0.0, "clustering.k-means");

const Core::ParameterInt KMeans::paramBatchSize_("batch-size", 4096, "clustering.k-means");

const Core::ParameterEnum KMeans::paramInitializationMode_("initialization-mode", "random",
		"random", "clustering.k-means");

KMeans::KMeans() :
		nClusters_(Core::Configuration::config(paramNumberOfClusters_)),
		meanInputFile_(Core::Configuration::config(paramMeanInputFile_)),
		meanOutputFile_(Core::Configuration::config(paramMeanOutputFile_)),
		maxIterations_(Core::Configuration::config(paramMaxIterations_)),
		stopThreshold_(Core::Configuration::config(paramStopThreshold_)),
		batchSize_(Core::Configuration::config(paramBatchSize_)),
		initializationMode_((InitializationMode)Core::Configuration::config(paramInitializationMode_)),
		featureDim_(0),
		nObservations_(0),
		initializedMeans_(false)
{}

void KMeans::writeParameters() {
	require(!meanOutputFile_.empty());
	means_.finishComputation();
	// save means
	means_.write(meanOutputFile_);
}

void KMeans::bufferFeature(const Math::Vector<Float>& f, u32 index) {
	require_eq(f.size(), featureDim_);
	for (u32 d = 0; d < featureDim_; d++)
		data_.at(d, index) = f.at(d);
}

u32 KMeans::nClusters() {
	return nClusters_;
}

void KMeans::randomInitialization() {
	means_.setToZero();
	Math::Random::initializeSRand();
	// generate random indices
	std::vector<u32> indices(nObservations_);
	for (u32 i = 0; i < nObservations_; i++) {
		indices[i] = i;
	}
	std::random_shuffle(indices.begin(), indices.end(), Math::Random::randomIntBelow);
	// make a reversely sorted  nClusters_ dimensional index array out of this
	indices.resize(nClusters_);
	std::sort(indices.begin(), indices.end());
	std::reverse(indices.begin(), indices.end());
	// assign the features with the selected indices to the clusters
	means_.finishComputation(false);
	u32 c = 0;
	for (u32 i = 0; i < featureReader_.totalNumberOfFeatures(); i++) {
		const Math::Vector<Float>& f = featureReader_.next();
		if (indices.back() == i) {
			indices.pop_back();
			for (u32 d = 0; d < featureDim_; d++) {
				means_.at(c, d) = f.at(d);
			}
			c++;
		}
	}
	means_.initComputation();
}

void KMeans::initializeSeeds() {
	switch ( initializationMode_ ) {
	case random:
	default:
		Core::Log::os("Initialize with random seeds.");
		randomInitialization();
	}
	initializedMeans_ = true;
}

void KMeans::distanceAndClusterAssignment() {
	batchClusterAssignment_.initComputation(false);
	dist_.initComputation(false);
	dist_.resize(nClusters_, data_.nColumns());
	batchClusterAssignment_.resize(data_.nColumns());
	dist_.setToZero();
	dist_.addMatrixProduct(means_, data_, (Float)1.0, (Float)1.0, false, false);
	tmp_.resize(nClusters_);
	tmp_.setToZero();
	tmp_.addSquaredSummedColumns(means_, (Float)-0.5);
	dist_.addToAllColumns(tmp_);
	// compute cluster assignment
	dist_.argMax(batchClusterAssignment_);
	batchClusterAssignment_.finishComputation();
	dist_.finishComputation();
}

void KMeans::generateBatch() {
	u32 i = 0;
	data_.finishComputation(false);
	data_.resize(featureDim_, batchSize_);
	while (featureReader_.hasFeatures() && (i < batchSize_)) {
		const Math::Vector<Float>& f = featureReader_.next();
		for (u32 d = 0; d < featureDim_; d++) {
			data_.at(d, i) = f.at(d);
		}
		i++;
	}
	data_.resize(featureDim_, i);
	data_.initComputation();
}

Float KMeans::clusterBatch(u32 startIndex, u32 endIndex) {
	distanceAndClusterAssignment();
	// store cluster assignment for batch
	for (u32 i = 0; i <= endIndex - startIndex; i++) {
		clusterAssignment_.at(i + startIndex) = batchClusterAssignment_.at(i);
	}
	// compute score of clustered batch
	Float score = data_.sumOfSquares();
	for (u32 i = 0; i <= endIndex - startIndex; i++) {
		score += dist_.at(batchClusterAssignment_.at(i), i) * (-2.0);
	}
	return score;
}

void KMeans::initialize() {
	// ensure the feature reader is initialized
	featureReader_.initialize();
	featureDim_ = featureReader_.featureDimension();
	nObservations_ = featureReader_.totalNumberOfFeatures();
	// does not work with shuffled features because the same feature order is required in each epoch
	require(featureReader_.shuffleBuffer() == false);

	// prepare vectors and matrices
	means_.initComputation();
	means_.resize(nClusters_, featureDim_);
	dist_.resize(nClusters_, batchSize_);
	tmp_.initComputation();
	batchClusterAssignment_.resize(batchSize_);
	clusterAssignment_.resize(nObservations_);
	clusterCount_.initComputation(false);
	clusterCount_.resize(nClusters_);

	// generate seeds for cluster means
	initializeSeeds();
}

void KMeans::generateClustering() {

	// compute clustering: loop over iterations and loop over features
	Float oldScore = Types::max<Float>();
	Float newScore = Types::max<Float>();
	Core::Log::openTag("clustering.k-means");

	/* kmeans main loop */
	u32 iteration = 0;
	for (iteration = 1; iteration <= maxIterations_; iteration++) {
		// cluster all observations
		u32 i = 0;
		newScore = 0;
		featureReader_.newEpoch();
		while (featureReader_.hasFeatures()) {
			generateBatch();
			newScore += clusterBatch(i, i + data_.nColumns() - 1);
			i += data_.nColumns();
		}
		// compute score of current clustering
		newScore /= nObservations_;
		// update means
		means_.setToZero();
		means_.finishComputation();
		clusterCount_.setToZero();
		clusterCount_.finishComputation();
		featureReader_.newEpoch();
		for (u32 i = 0; i < featureReader_.totalNumberOfFeatures(); i++) {
			const Math::Vector<Float>& f = featureReader_.next();
			for (u32 d = 0; d < featureDim_; d++) {
				means_.at(clusterAssignment_.at(i), d) += f.at(d);
			}
			clusterCount_.at(clusterAssignment_.at(i))++;
		}
		means_.initComputation();
		clusterCount_.initComputation();
		means_.divideRowsByScalars(clusterCount_);
		// finish iteration
		Core::Log::os("iteration ") << iteration - 1 << ": score=" << newScore;
		// exit loop in case of convergence
		if ((oldScore - newScore <= stopThreshold_) && (iteration > 1))
			break;
		oldScore = newScore;
	}
	Core::Log::closeTag();

	writeParameters();
}
