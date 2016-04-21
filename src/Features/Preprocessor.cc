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

#include "Preprocessor.hh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "../Core/Configuration.hh"
#include "../Core/Log.hh"
#include "../Core/Parameter.hh"
#include "../Core/Types.hh"
#include "../Core/Utils.hh"

using namespace Features;

/*
 * Preprocessor
 */
const Core::ParameterEnum Preprocessor::paramType_("type",
		"none, vector-subtraction, vector-division, matrix-multiplication, feature-selection, windowing, z-score, component-selection",
		"none", "features.preprocessor");

Preprocessor::Preprocessor(const char* name) :
		name_(name),
		inputDimension_(0),
		outputDimension_(0),
		isInitialized_(false)
{}

void Preprocessor::initialize(u32 inputDimension) {
	inputDimension_ = inputDimension;
	isInitialized_ = true;
}

u32 Preprocessor::inputDimension() const {
	require(isInitialized_);
	return inputDimension_;
}

u32 Preprocessor::outputDimension() const {
	require(isInitialized_);
	return outputDimension_;
}

Preprocessor* Preprocessor::createPreprocessor(const char* name) {
	Preprocessor* p = 0;
	switch ( (Type) Core::Configuration::config(paramType_, name)) {
	case vectorSubtraction:
		Core::Log::os() << "Create vector-subtraction preprocessor as " << name << ".";
		p = new VectorSubtractionPreprocessor(name);
		break;
	case vectorDivision:
		Core::Log::os() << "Create vector-division preprocessor as " << name << ".";
		p = new VectorDivisionPreprocessor(name);
		break;
	case matrixMultiplication:
		Core::Log::os() << "Create matrix-multiplication preprocessor as " << name << ".";
		p = new MatrixMultiplicationPreprocessor(name);
		break;
	case featureSelection:
		Core::Log::os() << "Create feature-selection preprocessor as " << name << ".";
		p = new FeatureSelectionPreprocessor(name);
		break;
	case windowing:
		Core::Log::os() << "Create windowing preprocessor as " << name << ".";
		p = new WindowingPreprocessor(name);
		break;
	case zscore:
		Core::Log::os() << "Create z-score preprocessor as " << name << ".";
		p = new ZScorePreprocessor(name);
		break;
	case componentSelection:
		Core::Log::os() << "Create component-selection preprocessor as " << name << ".";
		p = new ComponentSelectionPreprocessor(name);
		break;
	default:
		std::cerr << "Preprocessor::createPreprocessor: no type specified for preprocessor " << name << ". Abort." << std::endl;
		exit(1);
	}
	return p;
}

/*
 * VectorSubtractionPreprocessor
 */
const Core::ParameterString VectorSubtractionPreprocessor::paramVectorFile_("vector", "", "features.preprocessor");

VectorSubtractionPreprocessor::VectorSubtractionPreprocessor(const char* name) :
		Precursor(name),
		vectorFile_(Core::Configuration::config(paramVectorFile_, name_))
{}

void VectorSubtractionPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	require(!vectorFile_.empty());
	vector_.read(vectorFile_);
	require_eq(inputDimension_, vector_.nRows());
	outputDimension_ = inputDimension_;
}

void VectorSubtractionPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	out.copy(in);
	out.addToAllColumns(vector_, (Float)-1.0);
}

/*
 * VectorDivisionPreprocessor
 */
const Core::ParameterString VectorDivisionPreprocessor::paramVectorFile_("vector", "", "features.preprocessor");

VectorDivisionPreprocessor::VectorDivisionPreprocessor(const char* name) :
		Precursor(name),
		vectorFile_(Core::Configuration::config(paramVectorFile_, name_))
{}

void VectorDivisionPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	require(!vectorFile_.empty());
	vector_.read(vectorFile_);
	require_eq(inputDimension_, vector_.nRows());
	outputDimension_ = inputDimension_;
}

void VectorDivisionPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	out.copy(in);
	out.divideRowsByScalars(vector_);
}

/*
 * MatrixMultiplicationPreprocessor
 */
const Core::ParameterString MatrixMultiplicationPreprocessor::paramMatrixFile_("matrix", "", "features.preprocessor");

const Core::ParameterBool MatrixMultiplicationPreprocessor::paramTransposeMatrix_("transpose", true, "features.preprocessor");

MatrixMultiplicationPreprocessor::MatrixMultiplicationPreprocessor(const char* name) :
		Precursor(name),
		matrixFile_(Core::Configuration::config(paramMatrixFile_, name_)),
		transpose_(Core::Configuration::config(paramTransposeMatrix_, name_))
{}

void MatrixMultiplicationPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	require(!matrixFile_.empty());
	matrix_.read(matrixFile_, transpose_);
	require_eq(inputDimension_, matrix_.nColumns());
	outputDimension_ = matrix_.nRows();
}

void MatrixMultiplicationPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	// matrix multiplication
	out.addMatrixProduct(matrix_, in, 0.0, 1.0, false, false);
}

/*
 * FeatureSelectionPreprocessor
 */
const Core::ParameterInt FeatureSelectionPreprocessor::paramStartIndex_("selection-start-index", 0, "features.preprocessor");

const Core::ParameterInt FeatureSelectionPreprocessor::paramEndIndex_("selection-end-index", 0, "features.preprocessor");

FeatureSelectionPreprocessor::FeatureSelectionPreprocessor(const char* name) :
		Precursor(name),
		startIndex_(Core::Configuration::config(paramStartIndex_, name_)),
		endIndex_(Core::Configuration::config(paramEndIndex_, name_))
{
	require_ge(endIndex_ - startIndex_, 0);
}

void FeatureSelectionPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	require_gt(inputDimension_, endIndex_);
	outputDimension_ = endIndex_ - startIndex_ + 1;
}

void FeatureSelectionPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	for (u32 col = 0; col < out.nColumns(); col++) {
		for (u32 i = 0; i < out.nRows(); i++) {
			out.at(i, col) = in.at(i + startIndex_, col);
		}
	}
}

/*
 * WindowingPreprocessor
 */
const Core::ParameterInt WindowingPreprocessor::paramWindowSize_("window-size", 1, "features.preprocessor");

WindowingPreprocessor::WindowingPreprocessor(const char* name) :
		Precursor(name),
		windowSize_(Core::Configuration::config(paramWindowSize_, name_))
{
	if (windowSize_ % 2 == 0) {
		windowSize_++;
		Core::Log::os("WindowingPreprocessor: window-size is even. Set it to the next odd number (") << windowSize_ << ")";
	}
}

void WindowingPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension_ * windowSize_;
}

void WindowingPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	for (u32 col = 0; col < in.nColumns(); col++) {
		for (s32 w = (s32)-windowSize_ / 2; w <= (s32)windowSize_ / 2; w++) {
			u32 offset = inputDimension_ * (w + windowSize_ / 2);
			for (u32 i = 0; i < inputDimension_; i++) {
				if ( ((s32)col + w >= 0) && ((s32)col + w < (s32)in.nColumns()) )
					out.at(i + offset, col) = in.at(i, col + w);
				else
					out.at(i + offset, col) = 0;
			}
		}
	}
}

/*
 * ZScorePreprocessor
 */
ZScorePreprocessor::ZScorePreprocessor(const char* name) :
		Precursor(name)
{}

void ZScorePreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension_;
}

void ZScorePreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	// estimate mean
	Math::Vector<Float> mean(inputDimension_);
	mean.setToZero();
	mean.addSummedColumns(in, 1.0 / (Float)in.nColumns());
	// estimate standard deviation
	Math::Vector<Float> stdDev(inputDimension_);
	stdDev.setToZero();
	for (u32 n = 0; n < in.nColumns(); n++) {
		for (u32 d = 0; d < in.nRows(); d++) {
			stdDev.at(d) += (in.at(d, n) - mean.at(d)) * (in.at(d, n) - mean.at(d));
		}
	}
	stdDev.scale(1.0 / (Float)in.nColumns());
	for (u32 d = 0; d < stdDev.nRows(); d++) {
		stdDev.at(d) = sqrt(stdDev.at(d));
	}
	// normalize input
	out.copy(in);
	out.addToAllColumns(mean, -1.0);
	out.divideRowsByScalars(stdDev);
}

/*
 * ComponentSelectionPreprocessor
 */
const Core::ParameterBool ComponentSelectionPreprocessor::paramUseIndexList_("use-index-list", false, "features.preprocessor");

const Core::ParameterIntList ComponentSelectionPreprocessor::paramIndexList_("index-list", "", "features.preprocessor");

const Core::ParameterInt ComponentSelectionPreprocessor::paramFrom_("from", 0, "features.preprocessor");

const Core::ParameterInt ComponentSelectionPreprocessor::paramTo_("to", 0, "features.preprocessor");

ComponentSelectionPreprocessor::ComponentSelectionPreprocessor(const char* name) :
		Precursor(name),
		useIndexList_(Core::Configuration::config(paramUseIndexList_, name_)),
		indexList_(Core::Configuration::config(paramIndexList_, name_)),
		from_(Core::Configuration::config(paramFrom_, name_)),
		to_(Core::Configuration::config(paramTo_, name_))
{
	if (useIndexList_) {
		require_gt(indexList_.size(), 0);
	}
	else {
		require_ge(from_, 0);
		require_ge(to_, from_);
	}
}

void ComponentSelectionPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	if (useIndexList_) {
		for (u32 i = 0; i < indexList_.size(); i++) {
			require_ge(indexList_.at(i), 0);
			require_lt((u32)indexList_.at(i), inputDimension);
		}
		outputDimension_ = indexList_.size();
	}
	else {
		require_lt(to_, inputDimension);
		outputDimension_ = to_ + 1 - from_;
	}
}

void ComponentSelectionPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	if (useIndexList_) {
		for (u32 col = 0; col < out.nColumns(); col++) {
			for (u32 d = 0; d < indexList_.size(); d++) {
				out.at(d, col) = in.at((u32)indexList_.at(d), col);
			}
		}
	}
	else {
		out.copyBlockFromMatrix(in, from_, 0, 0, 0, outputDimension_, in.nColumns());
	}
}
