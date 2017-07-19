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
 * Preprocessor.cc
 *
 *  Created on: Sep 30, 2014
 *      Author: richard
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
#include <Math/Random.hh>

#ifdef MODULE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#else
namespace cv {
	typedef u32 Mat; // dummy definition if OpenCV is not used
}
#endif

using namespace Features;

/*
 * Preprocessor
 */
const Core::ParameterEnum Preprocessor::paramType_("type",
		"none, vector-subtraction, vector-division, matrix-multiplication, polynomial-expansion, windowing, window-pooling, "
		"z-score, l2-normalization, power-normalization, component-selection, random-image-cropping",
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
	case polynomialExpansion:
		Core::Log::os() << "Create polynomial-expansion preprocessor as " << name << ".";
		p = new PolynomialExpansionPreprocessor(name);
		break;
	case windowing:
		Core::Log::os() << "Create windowing preprocessor as " << name << ".";
		p = new WindowingPreprocessor(name);
		break;
	case windowPooling:
		Core::Log::os() << "Create window-pooling preprocessor as " << name << ".";
		p = new WindowPoolingPreprocessor(name);
		break;
	case zscore:
		Core::Log::os() << "Create z-score preprocessor as " << name << ".";
		p = new ZScorePreprocessor(name);
		break;
	case l2normalization:
		Core::Log::os() << "Create l2-normalization preprocessor as " << name << ".";
		p = new L2NormalizationPreprocessor(name);
		break;
	case powerNormalization:
		Core::Log::os() << "Create power-normalization preprocessor as " << name << ".";
		p = new PowerNormalizationPreprocessor(name);
		break;
	case componentSelection:
		Core::Log::os() << "Create component-selection preprocessor as " << name << ".";
		p = new ComponentSelectionPreprocessor(name);
		break;
	case randomImageCropping:
		Core::Log::os() << "Create random-image-cropping preprocessor as " << name << ".";
		p = new RandomImageCroppingPreprocessor(name);
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
 * PolynomialExpansionPreprocessor
 */
const Core::ParameterInt PolynomialExpansionPreprocessor::paramOrder_("order", 1, "features.preprocessor");

const Core::ParameterBool PolynomialExpansionPreprocessor::paramOnlyDiagonal_("only-diagonal", true, "features.preprocessor");

PolynomialExpansionPreprocessor::PolynomialExpansionPreprocessor(const char* name) :
		Precursor(name),
		order_(Core::Configuration::config(paramOrder_, name_)),
		onlyDiagonal_(Core::Configuration::config(paramOnlyDiagonal_, name_))
{
	require_ge(order_, 1);
}

void PolynomialExpansionPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension;
	if (order_ > 1)
		outputDimension_ += onlyDiagonal_ ? inputDimension : inputDimension * (inputDimension + 1) / 2;
	if (order_ > 2)
		outputDimension_ += onlyDiagonal_ ? inputDimension : inputDimension * (inputDimension + 1) * (inputDimension + 2) / 6;
	if (order_ > 3) {
		Core::Error::msg("PolynomialExpansionPreprocessor: polynomial order > 3 not implemented.") << Core::Error::abort;
	}
}

void PolynomialExpansionPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());

	if (order_ == 1)
		out.copy(in);
	if ((order_ == 2) && onlyDiagonal_)
		out.setToDiagonalSecondOrderFeatures(in);
	if ((order_ == 2) && (!onlyDiagonal_))
		out.setToSecondOrderFeatures(in);
	if ((order_ == 3) && onlyDiagonal_)
		out.setToDiagonalThirdOrderFeatures(in);
	if ((order_ == 3) && (!onlyDiagonal_))
		out.setToThirdOrderFeatures(in);
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
 * WindowPoolingPreprocessor
 */
const Core::ParameterInt WindowPoolingPreprocessor::paramWindowSize_("window-size", 1, "features.preprocessor");

WindowPoolingPreprocessor::WindowPoolingPreprocessor(const char* name) :
		Precursor(name),
		windowSize_(Core::Configuration::config(paramWindowSize_, name_))
{
	if (windowSize_ % 2 == 0) {
		windowSize_++;
		Core::Log::os("WindowPoolingPreprocessor: window-size is even. Set it to the next odd number (") << windowSize_ << ")";
	}
}

void WindowPoolingPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension_;
}

void WindowPoolingPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.resize(outputDimension_, in.nColumns());
	out.setToZero();
	for (u32 col = 0; col < in.nColumns(); col++) {
		for (s32 w = (s32)-windowSize_ / 2; w <= (s32)windowSize_ / 2; w++) {
			for (u32 d = 0; d < inputDimension_; d++) {
				out.at(d, col) += in.at(d, std::min(std::max((s32)col + w, 0), (s32)in.nColumns() - 1));
			}
		}
	}
	out.scale(1.0 / windowSize_);
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
 * L2NormalizationPreprocessor
 */
L2NormalizationPreprocessor::L2NormalizationPreprocessor(const char* name) :
		Precursor(name)
{}

void L2NormalizationPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension_;
}

void L2NormalizationPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	Math::Vector<Float> l2Norm(in.nColumns());
	l2Norm.columnwiseInnerProduct(in, in);
	l2Norm.signedPow(0.5);
	l2Norm.ensureMinimalValue(Types::absMin<Float>()); // prevent division by 0
	out.resize(outputDimension_, in.nColumns());
	out.copy(in);
	out.divideColumnsByScalars(l2Norm);
}

/*
 * PowerNormalizationPreprocessor
 */
const Core::ParameterFloat PowerNormalizationPreprocessor::paramPower_("power", 0.5, "features.preprocessor");

PowerNormalizationPreprocessor::PowerNormalizationPreprocessor(const char* name) :
		Precursor(name),
		power_(Core::Configuration::config(paramPower_, name))
{
	require_gt(power_, 0);
}

void PowerNormalizationPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	outputDimension_ = inputDimension_;
}

void PowerNormalizationPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	require(isInitialized_);
	require_eq(inputDimension_, in.nRows());
	out.copy(in);
	out.signedPow(power_);
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

/*
 * RandomImageCroppingPreprocessor
 */
const Core::ParameterInt RandomImageCroppingPreprocessor::paramInputWidth_("input-width", 340, "features.preprocessor");

const Core::ParameterInt RandomImageCroppingPreprocessor::paramInputHeight_("input-height", 256, "features.preprocessor");

const Core::ParameterInt RandomImageCroppingPreprocessor::paramChannels_("channels", 3, "features.preprocessor");

const Core::ParameterIntList RandomImageCroppingPreprocessor::paramPossibleCropSideLengths_("possible-side-lengths",
		"256, 224, 192, 168", "features.preprocessor");

const Core::ParameterInt RandomImageCroppingPreprocessor::paramCropWidth_("crop-width", 224, "features.preprocessor");

const Core::ParameterInt RandomImageCroppingPreprocessor::paramCropHeight_("crop-height", 224, "features.preprocessor");

RandomImageCroppingPreprocessor::RandomImageCroppingPreprocessor(const char* name) :
		Precursor(name),
		inputWidth_(Core::Configuration::config(paramInputWidth_, name_)),
		inputHeight_(Core::Configuration::config(paramInputHeight_, name_)),
		channels_(Core::Configuration::config(paramChannels_, name_)),
		cropWidth_(Core::Configuration::config(paramCropWidth_, name_)),
		cropHeight_(Core::Configuration::config(paramCropHeight_, name_))
{
	std::vector<s32> possibleCropSideLengths = Core::Configuration::config(paramPossibleCropSideLengths_, name_);
	for (u32 i = 0; i < possibleCropSideLengths.size(); i++) {
		require_le(0, possibleCropSideLengths.at(i));
		sideLengthList_.push_back((u32)possibleCropSideLengths.at(i));
	}
}

void RandomImageCroppingPreprocessor::initialize(u32 inputDimension) {
	Precursor::initialize(inputDimension);
	require_eq(inputWidth_ * inputHeight_ * channels_, inputDimension);
	outputDimension_ = cropWidth_ * cropHeight_ * channels_;
}

void RandomImageCroppingPreprocessor::createCrop(const Math::Matrix<Float>& in, Math::Matrix<Float>& out, u32 column) {
	/* horizontal flip? */
	bool flip = ( Math::Random::random() < 0.5 ? true : false);
	/* determine size and position of crop */
	u32 cropWidth = std::min( sideLengthList_.at(Math::Random::randomInt(0, sideLengthList_.size() - 1)), inputWidth_);
	u32 cropHeight = std::min( sideLengthList_.at(Math::Random::randomInt(0, sideLengthList_.size() - 1)), inputHeight_);
	// center crop (20% probability)
	u32 posX = (inputWidth_ - cropWidth) / 2;
	u32 posY = (inputHeight_ - cropHeight) / 2;
	// corner crop (80% probability, 20% for each corner)
	if (Math::Random::random() < 0.8) {
		posX = Math::Random::randomInt(0, 1) * (inputWidth_ - cropWidth);
		posY = Math::Random::randomInt(0, 1) * (inputHeight_ - cropHeight);
	}
	/* create crop */
#ifdef MODULE_OPENCV
	/* column to image as cv::Mat */
	cv::Mat img(inputHeight_, inputWidth_, (channels_ == 3 ? CV_32FC3 : CV_32FC1));
	Core::Utils::copyMemoryToCVMat(in.begin() + inputDimension_ * column, img);
	/* cropping */
	img = img(cv::Rect(posX, posY, cropWidth, cropHeight));
	cv::resize(img, img, cv::Size(cropWidth_, cropHeight_));
	/* horizontal flipping */
	if (flip)
		cv::flip(img, img, 1);
	/* write result to out */
	Core::Utils::copyCVMatToMemory(img, out.begin() + outputDimension_ * column);
#else
	Core::Error::msg("RandomImageCroppingPreprocessor::work requires OpenCV but binary is not compiled with OpenCV support.") << Core::Error::abort;
#endif
}

void RandomImageCroppingPreprocessor::work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) {
	out.resize(outputDimension_, in.nColumns());
	for (u32 col = 0; col < in.nColumns(); col++) {
		createCrop(in, out, col);
	}
}


