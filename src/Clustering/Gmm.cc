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
 * GmmTrainer.cc
 *
 *  Created on: Jan 13, 2015
 *      Author: richard
 */

#include "Gmm.hh"

using namespace Clustering;

/*
 * GmmBase
 */
const Core::ParameterString GmmBase::paramMeanInputFile_("mean-input-file", "", "clustering.gaussian-mixture-model");

const Core::ParameterString GmmBase::paramVarianceInputFile_("variance-input-file", "", "clustering.gaussian-mixture-model");

const Core::ParameterString GmmBase::paramWeightsInputFile_("weights-input-file", "", "clustering.gaussian-mixture-model");

const Core::ParameterString GmmBase::paramMeanOutputFile_("mean-output-file", "", "clustering.gaussian-mixture-model");

const Core::ParameterString GmmBase::paramVarianceOutputFile_("variance-output-file", "", "clustering.gaussian-mixture-model");

const Core::ParameterString GmmBase::paramWeightsOutputFile_("weights-output-file", "", "clustering.gaussian-mixture-model");

GmmBase::GmmBase() :
		meanInputFile_(Core::Configuration::config(paramMeanInputFile_)),
		varianceInputFile_(Core::Configuration::config(paramVarianceInputFile_)),
		weightsInputFile_(Core::Configuration::config(paramWeightsInputFile_)),
		meanOutputFile_(Core::Configuration::config(paramMeanOutputFile_)),
		varianceOutputFile_(Core::Configuration::config(paramVarianceOutputFile_)),
		weightsOutputFile_(Core::Configuration::config(paramWeightsOutputFile_))
{}

/*
 * GmmTrainer
 */
const Core::ParameterFloat GmmTrainer::paramMinVariance_("minimal-variance", 1e-07, "clustering.gaussian-mixture-model");

const Core::ParameterInt GmmTrainer::paramMaxIterations_("iterations", 10, "clustering.gaussian-mixture-model");

const Core::ParameterInt GmmTrainer::paramNumberOfDensities_("number-of-densities", 1, "clustering.gaussian-mixture-model");

const Core::ParameterInt GmmTrainer::paramBatchSize_("batch-size", 4096, "clustering.gaussian-mixture-model");

const Core::ParameterBool GmmTrainer::paramMaximumApproximation_("maximum-approximation", false, "clustering.gaussian-mixture-model");

GmmTrainer::GmmTrainer() :
		minVariance_(Core::Configuration::config(paramMinVariance_)),
		maxIterations_(Core::Configuration::config(paramMaxIterations_)),
		nMixtures_(Core::Configuration::config(paramNumberOfDensities_)),
		batchSize_(Core::Configuration::config(paramBatchSize_)),
		maximumApproximation_(Core::Configuration::config(paramMaximumApproximation_)),
		logLik_(Types::max<Float>())
{}

void GmmTrainer::initializeParameters() {
	/* initialize mean */
	if (!meanInputFile_.empty()) {
		Core::Log::os("load means from ") << meanInputFile_;
		oldMu_.read(meanInputFile_, true);
		require_eq(featureReader_.featureDimension(), oldMu_.nRows());
		require_eq(nMixtures_, oldMu_.nColumns());
	} else {
		Core::Log::os("initialize means randomly");
		oldMu_.resize(featureReader_.featureDimension(), nMixtures_);
		Math::Random::initializeSRand();
		// generate random indices
		std::vector<u32> indices(featureReader_.totalNumberOfFeatures());
		for (u32 i = 0; i < featureReader_.totalNumberOfFeatures(); i++) {
			indices[i] = i;
		}
		std::random_shuffle(indices.begin(), indices.end(), Math::Random::randomIntBelow);
		// make a reversely sorted index array out of this
		indices.resize(nMixtures_);
		std::sort(indices.begin(), indices.end());
		std::reverse(indices.begin(), indices.end());
		// assign the features with the selected indices to the clusters
		u32 c = 0;
		for (u32 i = 0; i < featureReader_.totalNumberOfFeatures(); i++) {
			const Math::Vector<Float>& f = featureReader_.next();
			if (indices.back() == i) {
				indices.pop_back();
				for (u32 d = 0; d < featureReader_.featureDimension(); d++) {
					oldMu_.at(d, c) = f.at(d);
				}
				c++;
			}
		}
	}
	/* initialize variance */
	if (!varianceInputFile_.empty()) {
		Core::Log::os("load variances from ") << varianceInputFile_;
		oldSigma_.read(varianceInputFile_, true);
		require_eq(featureReader_.featureDimension(), oldSigma_.nRows());
		require_eq(nMixtures_, oldSigma_.nColumns());
	} else {
		oldSigma_.resize(featureReader_.featureDimension(), nMixtures_);
		oldSigma_.initComputation();
		oldSigma_.fill(1);
	}
	/* initialize weights */
	if (!weightsInputFile_.empty()) {
		Core::Log::os("load weights from ") << weightsInputFile_;
		oldWeights_.read(weightsInputFile_);
		require_eq(nMixtures_, oldWeights_.nRows());
	} else {
		oldWeights_.resize(nMixtures_);
		oldWeights_.initComputation();
		oldWeights_.fill(1.0 / nMixtures_);
	}
}

void GmmTrainer::initialize() {
	featureReader_.initialize();
	// initialize parameters
	initializeParameters();
	// set to computation mode
	oldMu_.initComputation();
	oldSigma_.initComputation();
	oldWeights_.initComputation();
	newMu_.initComputation();
	newSigma_.initComputation();
	newWeights_.initComputation();
	posteriors_.initComputation();
	lambda_.initComputation();
	bias_.initComputation();
	tmpMatrix_.initComputation();
	tmpVector_.initComputation();
	// resize
	newMu_.resize(featureReader_.featureDimension(), nMixtures_);
	newSigma_.resize(featureReader_.featureDimension(), nMixtures_);
	newWeights_.resize(nMixtures_);
	lambda_.resize(2 * featureReader_.featureDimension(), nMixtures_);
	bias_.resize(nMixtures_);
}

void GmmTrainer::writeParameters() {
	require(!meanOutputFile_.empty());
	require(!varianceOutputFile_.empty());
	require(!weightsOutputFile_.empty());
	oldMu_.finishComputation();
	oldSigma_.finishComputation();
	oldWeights_.finishComputation();
	oldMu_.write(meanOutputFile_, true);
	oldSigma_.write(varianceOutputFile_, true);
	oldWeights_.write(weightsOutputFile_);
}

void GmmTrainer::transformForSoftmax() {
	bias_.setToZero();
	lambda_.finishComputation(false);
	bias_.finishComputation();
	oldMu_.finishComputation();
	oldSigma_.finishComputation();
	oldWeights_.finishComputation();

	for (u32 c = 0; c < nMixtures_; c++) {
		for (u32 d = 0; d < featureReader_.featureDimension(); d++) {
			lambda_.at(d, c) = oldMu_.at(d, c) / oldSigma_.at(d, c);
			lambda_.at(d + featureReader_.featureDimension(), c) = -0.5 / oldSigma_.at(d, c);
			bias_.at(c) += pow(oldMu_.at(d, c), 2) / oldSigma_.at(d, c) + log(oldSigma_.at(d, c));
		}
		bias_.at(c) = log(oldWeights_.at(c)) -0.5 * (bias_.at(c) + featureReader_.featureDimension() * log(2*M_PI));
	}

	bias_.initComputation();
	lambda_.initComputation();
	oldMu_.initComputation(false);
	oldSigma_.initComputation(false);
	oldWeights_.initComputation(false);
}

void GmmTrainer::computePosteriors() {
	tmpMatrix_.resize(buffer_.nRows() * 2, buffer_.nColumns());
	tmpMatrix_.setToDiagonalSecondOrderFeatures(buffer_);
	posteriors_.resize(nMixtures_, tmpMatrix_.nColumns());
	posteriors_.setToZero();
	posteriors_.addMatrixProduct(lambda_, tmpMatrix_, 0, 1, true, false);
	posteriors_.addToAllColumns(bias_);

	// compute log likelihood score in a numerically stable way
	if (maximumApproximation_) {
		tmpVector_.resize(buffer_.nColumns());
		tmpVector_.getMaxOfColumns(posteriors_);
		logLik_ += tmpVector_.sum();
	}
	else {
		tmpVector_.resize(buffer_.nColumns());
		tmpVector_.setToZero();
		tmpMatrix_.resize(posteriors_.nRows(), posteriors_.nColumns());
		tmpMatrix_.copy(posteriors_);
		tmpVector_.swap(tmpMatrix_);
		Float max = tmpVector_.get(tmpVector_.argMax());
		tmpVector_.swap(tmpMatrix_);
		tmpMatrix_.resize(posteriors_.nRows(), posteriors_.nColumns());
		tmpMatrix_.addConstantElementwise(-max);
		tmpMatrix_.exp();
		tmpVector_.addSummedRows(tmpMatrix_);
		tmpVector_.log();
		logLik_ += tmpVector_.sum() + tmpVector_.size() * max;
	}

	if (maximumApproximation_)
		posteriors_.max();
	else
		posteriors_.softmax();
}

void GmmTrainer::updateParameters() {
	computePosteriors();
	newWeights_.addSummedColumns(posteriors_);
	newMu_.addMatrixProduct(buffer_, posteriors_, 1, 1, false, true);
	buffer_.elementwiseMultiplication(buffer_);
	newSigma_.addMatrixProduct(buffer_, posteriors_, 1, 1, false, true);
}

void GmmTrainer::finalizeEstimation() {
	newMu_.divideColumnsByScalars(newWeights_);
	newSigma_.divideColumnsByScalars(newWeights_);
	tmpMatrix_.resize(newMu_.nRows(), newMu_.nColumns());
	tmpMatrix_.copy(newMu_);
	tmpMatrix_.elementwiseMultiplication(tmpMatrix_);
	newSigma_.add(tmpMatrix_, (Float)-1);
	newWeights_.scale(1.0 / featureReader_.totalNumberOfFeatures());
	newSigma_.ensureMinimalValue(minVariance_);
}

void GmmTrainer::bufferFeatures() {
	buffer_.finishComputation(false);
	buffer_.resize(featureReader_.featureDimension(), batchSize_);
	u32 nBufferedFeatures = 0;
	while ((featureReader_.hasFeatures()) && (nBufferedFeatures < batchSize_)) {
		const Math::Vector<Float>& f = featureReader_.next();
		for (u32 i = 0; i < f.size(); i++) {
			buffer_.at(i, nBufferedFeatures) = f.at(i);
		}
		nBufferedFeatures++;
	}
	buffer_.safeResize(buffer_.nRows(), nBufferedFeatures);
	buffer_.initComputation();
}

void GmmTrainer::generateClustering() {
	Core::Log::openTag("gaussian-mixture-model-generation");
	Float oldLogLik = Types::min<Float>();
	for (u32 iter = 0; iter < maxIterations_; iter++) {
		Core::Log::os("Start iteration ") << iter + 1;
		newMu_.setToZero();
		newSigma_.setToZero();
		newWeights_.setToZero();
		transformForSoftmax();
		logLik_ = 0;
		featureReader_.newEpoch();
		while (featureReader_.hasFeatures()) {
			bufferFeatures();
			updateParameters();
		}
		finalizeEstimation();

		Core::Log::os("log-likelihood score: ") << logLik_;
		oldMu_.swap(newMu_);
		oldSigma_.swap(newSigma_);
		oldWeights_.swap(newWeights_);

		//TODO: implement as two-pass algorithm for numerical stability

		logLik_ /= featureReader_.totalNumberOfFeatures();
		if ((logLik_ > Types::min<Float>()) && (logLik_ < Types::max<Float>()) && (oldLogLik >= logLik_)) {
			break;
		}
		oldLogLik = logLik_;
	}
	Core::Log::closeTag();
	writeParameters();
}

/*
 * GmmDensitySplitter
 */

void GmmDensitySplitter::readParameters() {
	require(!meanInputFile_.empty());
	require(!varianceInputFile_.empty());
	require(!weightsInputFile_.empty());
	oldMu_.read(meanInputFile_, true);
	oldSigma_.read(varianceInputFile_, true);
	oldWeights_.read(weightsInputFile_);
}

void GmmDensitySplitter::writeParameters() {
	require(!meanOutputFile_.empty());
	require(!varianceOutputFile_.empty());
	require(!weightsOutputFile_.empty());
	// save new parameters in scientific format
	newMu_.write(meanOutputFile_, true, true);
	newSigma_.write(varianceOutputFile_, true, true);
	newWeights_.write(weightsOutputFile_, true);
}

void GmmDensitySplitter::split() {
	readParameters();
	// split means
	// TODO: shift in direction of largest variance
	oldMu_.scale(1.0001);
	newMu_.resize(oldMu_.nRows(), 2 * oldMu_.nColumns());
	newMu_.copyBlockFromMatrix(oldMu_, 0, 0, 0, 0, oldMu_.nRows(), oldMu_.nColumns());
	oldMu_.scale(0.999 / 1.0001);
	newMu_.copyBlockFromMatrix(oldMu_, 0, 0, 0, oldMu_.nColumns(), oldMu_.nRows(), oldMu_.nColumns());
	// split variances
	newSigma_.resize(oldSigma_.nRows(), 2 * oldSigma_.nColumns());
	newSigma_.copyBlockFromMatrix(oldSigma_, 0, 0, 0, 0, oldSigma_.nRows(), oldSigma_.nColumns());
	newSigma_.copyBlockFromMatrix(oldSigma_, 0, 0, 0, oldSigma_.nColumns(), oldSigma_.nRows(), oldSigma_.nColumns());
	// split weights
	newWeights_.resize(2 * oldWeights_.nRows());
	Math::Matrix<Float> tmp1;
	Math::Matrix<Float> tmp2;
	tmp1.swap(oldWeights_);
	tmp2.swap(newWeights_);
	tmp2.copyBlockFromMatrix(tmp1, 0, 0, 0, 0, tmp1.nRows(), tmp1.nColumns());
	tmp2.copyBlockFromMatrix(tmp1, 0, 0, tmp1.nRows(), 0, tmp1.nRows(), tmp1.nColumns());
	tmp2.swap(newWeights_);
	newWeights_.scale(0.5);
	writeParameters();
}
