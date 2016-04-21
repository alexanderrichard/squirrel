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

#include "PrincipalComponentAnalysis.hh"
#include <Math/Lapack.hh>
#include <sstream>

using namespace FeatureTransformation;

/*
 * PrincipalComponentAnalysis
 */
const Core::ParameterString PrincipalComponentAnalysis::paramMeanFile_("mean-file", "",
		"feature-transformation.principal-component-analysis");

const Core::ParameterString PrincipalComponentAnalysis::paramPcaMatrixFile_("pca-matrix-file", "",
		"feature-transformation.principal-component-analysis");

const Core::ParameterInt PrincipalComponentAnalysis::paramNumberOfPrincipalComponents_("number-of-principal-components",
		Types::max<u32>(), "feature-transformation.principal-component-analysis");

const Core::ParameterBool PrincipalComponentAnalysis::paramWhitening_("whitening",
		true, "feature-transformation.principal-component-analysis");

const Core::ParameterBool PrincipalComponentAnalysis::paramEstimateMean_("estimate-mean", true,
		"feature-transformation.principal-component-analysis");

const Core::ParameterEnum PrincipalComponentAnalysis::paramDecompositionMethod_("decomposition-method", "eigendecomposition, svd",
		"eigendecomposition", "feature-transformation.principal-component-analysis");

PrincipalComponentAnalysis::PrincipalComponentAnalysis() :
		decompositionMethod_((DecompositionMethod) Core::Configuration::config(paramDecompositionMethod_)),
		meanFile_(Core::Configuration::config(paramMeanFile_)),
		pcaMatrixFile_(Core::Configuration::config(paramPcaMatrixFile_)),
		nPrincipalComponents_(Core::Configuration::config(paramNumberOfPrincipalComponents_)),
		whitening_(Core::Configuration::config(paramWhitening_)),
		estimateMean_(Core::Configuration::config(paramEstimateMean_))
{
	require_gt(nPrincipalComponents_, 0);
}

void PrincipalComponentAnalysis::initialize() {
	featureReader_.initialize();

	if (!estimateMean_) {
		require(!meanFile_.empty());
		mean_.read(meanFile_);
		require_eq(mean_.nRows(), featureReader_.featureDimension());
	}
	else {
		mean_.resize(featureReader_.featureDimension());
		mean_.setToZero();
	}

	if (nPrincipalComponents_ > featureReader_.featureDimension()) {
		nPrincipalComponents_ = featureReader_.featureDimension();
	}
}

void PrincipalComponentAnalysis::estimateMean() {
	// assume featureReader_ is at beginning of an epoch
	while (featureReader_.hasFeatures()) {
		mean_.add(featureReader_.next());
	}
	mean_.scale((Float) 1.0 / featureReader_.totalNumberOfFeatures());
	// write mean to file
	if (!meanFile_.empty())
		mean_.write(meanFile_);
}

void PrincipalComponentAnalysis::estimateScatterMatrix() {
	dataMatrix_.resize(featureReader_.featureDimension(), featureReader_.featureDimension());
	dataMatrix_.setToZero();
	while (featureReader_.hasFeatures()) {
		Math::Vector<Float> tmp(featureReader_.featureDimension());
		tmp.copy(featureReader_.next());
		tmp.add(mean_, (Float)-1.0);
		dataMatrix_.addOuterProduct(tmp, tmp, (Float)1.0);
	}
}

void PrincipalComponentAnalysis::computeDataMatrix() {
	dataMatrix_.resize(featureReader_.featureDimension(), featureReader_.totalNumberOfFeatures());
	while (featureReader_.hasFeatures()) {
		for (u32 n = 0; n < featureReader_.totalNumberOfFeatures(); n++) {
			const Math::Vector<Float>& tmp = featureReader_.next();
			for (u32 d = 0; d < featureReader_.featureDimension(); d++) {
				dataMatrix_.at(d, n) = tmp.at(d) - mean_.at(d);
			}
		}
	}
}

void PrincipalComponentAnalysis::eigenDecomposition() {
	// scale dataMatrix (in this case: scatter matrix) by number of features
	dataMatrix_.scale((Float) 1.0 / featureReader_.totalNumberOfFeatures());

	u32 firstEigenvalueIndex = dataMatrix_.nRows() - nPrincipalComponents_ + 1;
	u32 lastEigenvalueIndex = dataMatrix_.nRows();

	// eigenvalue decomposition
	eigenvalues_.resize(featureReader_.featureDimension());
	eigenvalues_.setToZero();
	eigenvectors_.resize(featureReader_.featureDimension(), featureReader_.featureDimension());
	eigenvectors_.setToZero();
	Math::Vector<u32> support(2 * featureReader_.featureDimension());
	support.setToZero();

	u32 result = Math::syevr(dataMatrix_.nRows(), (Float*) dataMatrix_.begin(),
			firstEigenvalueIndex, lastEigenvalueIndex,
			(Float*) eigenvalues_.begin(), (Float*) eigenvectors_.begin(), (int*) support.begin());

	if (result != 0) {
		std::cerr << "PrincipalComponentAnalysis::eigenDecomposition: Eigendecomposition failed. Abort." << std::endl;
		exit(1);
	}

	eigenvalues_.safeResize(nPrincipalComponents_);
	eigenvectors_.safeResize(eigenvectors_.nRows(), nPrincipalComponents_);

	// revert eigenvalue list to be in descending order
	for (u32 i = 0; i < eigenvalues_.size() / 2; i++) {
		std::swap(eigenvalues_.at(i), eigenvalues_.at(eigenvalues_.size() - 1 - i));
	}
	// revert columns of eigenvector matrix accordingly
	for (u32 i = 0; i < eigenvectors_.nColumns() / 2; i++) {
		Math::Vector<Float> tmp(eigenvectors_.nRows());
		eigenvectors_.getColumn(i, tmp);
		for (u32 j = 0; j < eigenvectors_.nRows(); j++) {
			eigenvectors_.at(j, i) = eigenvectors_.at(j, eigenvectors_.nColumns() - 1 - i);
			eigenvectors_.at(j, eigenvectors_.nColumns() - 1 - i) = tmp.at(j);
		}
	}
}

void PrincipalComponentAnalysis::singularValueDecomposition() {
	// singular value decomposition
	eigenvalues_.resize(std::min(dataMatrix_.nRows(), dataMatrix_.nColumns()));
	eigenvalues_.setToZero();
	eigenvectors_.resize(dataMatrix_.nRows(), std::min(dataMatrix_.nRows(), dataMatrix_.nColumns()));
	eigenvectors_.setToZero();

	u32 result = Math::gesvd(dataMatrix_.nRows(), dataMatrix_.nColumns(),
			dataMatrix_.begin(), eigenvalues_.begin(), eigenvectors_.begin());

	if (result != 0) {
		std::cerr << "PrincipalComponentAnalysis::singularValueDecomposition: SVD failed. Abort." << std::endl;
		exit(1);
	}

	eigenvalues_.safeResize(nPrincipalComponents_);
	eigenvectors_.safeResize(eigenvectors_.nRows(), nPrincipalComponents_);

	eigenvalues_.elementwiseMultiplication(eigenvalues_);
	// scale eigenvalues according to number of features
	// (corresponds to using covariance matrix instead of scatter matrix for eigendecomposition)
	eigenvalues_.scale((Float) 1.0 / dataMatrix_.nColumns());
}

void PrincipalComponentAnalysis::estimatePca() {

	if (decompositionMethod_ == eigen) {
		eigenDecomposition();
	}
	else {
		singularValueDecomposition();
	}

	// print nPrincipalComponents_ largest eigenvalues in ascending order
	Core::Log::openTag("largest-eigenvalues");
	std::stringstream s;
	for (u32 i = 0; i < nPrincipalComponents_; i++) {
		s << eigenvalues_.at(i) << " ";
	}
	Core::Log::os() << s.str();

	// whitening
	if (whitening_) {
		for (u32 i = 0; i < eigenvalues_.nRows(); i++) {
			eigenvalues_.ensureMinimalValue(0);
			eigenvalues_.at(i) = sqrt(eigenvalues_.at(i));
			for (u32 j = 0; j < eigenvectors_.nRows(); j++) {
				if (eigenvalues_.at(i) == 0) {
					eigenvectors_.at(j, i) = (i == j ? 1 : 0);
				}
				else {
					eigenvectors_.at(j, i) = eigenvectors_.at(j, i) / eigenvalues_.at(i);
				}
			}
		}
	}

	Core::Log::closeTag();

	if (!pcaMatrixFile_.empty())
		eigenvectors_.write(pcaMatrixFile_);
}

void PrincipalComponentAnalysis::estimate() {
	Core::Log::openTag("principal-component-analysis");
	if (decompositionMethod_ == eigen) {
		Core::Log::os("Use eigenvalue decomposition.");
	}
	else {
		Core::Log::os("Use singular value decomposition.");
	}

	if (estimateMean_) {
		estimateMean();
		featureReader_.newEpoch();
	}
	if (decompositionMethod_ == eigen) {
		estimateScatterMatrix();
	}
	else {
		computeDataMatrix();
	}

	estimatePca();

	Core::Log::closeTag();
}
