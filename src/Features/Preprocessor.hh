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
 * Preprocessor.hh
 *
 *  Created on: Sep 30, 2014
 *      Author: richard
 */

#ifndef FEATURES_PREPROCESSOR_HH_
#define FEATURES_PREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

namespace Features {

/*
 * Base class for feature preprocessors
 */
class Preprocessor
{
private:
	static const Core::ParameterEnum paramType_;
	enum Type { none, vectorSubtraction, vectorDivision, matrixMultiplication, polynomialExpansion, windowing, windowPooling,
		zscore, l2normalization, powerNormalization, componentSelection, randomImageCropping };
protected:
	std::string name_;
	u32 inputDimension_;
	u32 outputDimension_;
	bool isInitialized_;
public:
	Preprocessor(const char* name);
	virtual ~Preprocessor() {}
	virtual void initialize(u32 inputDimension);
	u32 inputDimension() const;
	u32 outputDimension() const;
	virtual bool needsContext() { return false; }

	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out) = 0;

	static Preprocessor* createPreprocessor(const char* name);
};

/*
 * vector subtraction
 */
class VectorSubtractionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramVectorFile_;
protected:
	std::string vectorFile_;
	Math::Vector<Float> vector_;
public:
	VectorSubtractionPreprocessor(const char* name);
	virtual ~VectorSubtractionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * vector division
 */
class VectorDivisionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramVectorFile_;
protected:
	std::string vectorFile_;
	Math::Vector<Float> vector_;
public:
	VectorDivisionPreprocessor(const char* name);
	virtual ~VectorDivisionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * matrix multiplication
 */
class MatrixMultiplicationPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterString paramMatrixFile_;
	static const Core::ParameterBool paramTransposeMatrix_;
protected:
	std::string matrixFile_;
	Math::Matrix<Float> matrix_;
	bool transpose_;
public:
	MatrixMultiplicationPreprocessor(const char* name);
	virtual ~MatrixMultiplicationPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * polynomial feature expansion
 */
class PolynomialExpansionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterInt paramOrder_;
	static const Core::ParameterBool paramOnlyDiagonal_;
protected:
	u32 order_;
	bool onlyDiagonal_;
public:
	PolynomialExpansionPreprocessor(const char* name);
	virtual ~PolynomialExpansionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * windowing (concatenate neighboring features within a sequence)
 */
class WindowingPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterInt paramWindowSize_;
protected:
	u32 windowSize_;
public:
	WindowingPreprocessor(const char* name);
	virtual ~WindowingPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual bool needsContext() { return true; }
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * window-pooling (average pooling of neighboring features within a sequence)
 */
class WindowPoolingPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterInt paramWindowSize_;
protected:
	u32 windowSize_;
public:
	WindowPoolingPreprocessor(const char* name);
	virtual ~WindowPoolingPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual bool needsContext() { return true; }
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * z-score normalization on sequence level
 */
class ZScorePreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
public:
	ZScorePreprocessor(const char* name);
	virtual ~ZScorePreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual bool needsContext() { return true; }
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * l2-normalization on feature vector level
 */
class L2NormalizationPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
public:
	L2NormalizationPreprocessor(const char* name);
	virtual ~L2NormalizationPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * power-normalization on feature vector level
 */
class PowerNormalizationPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterFloat paramPower_;
protected:
	Float power_;
public:
	PowerNormalizationPreprocessor(const char* name);
	virtual ~PowerNormalizationPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * select specific components from each input vector
 */
class ComponentSelectionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterBool paramUseIndexList_;
	static const Core::ParameterIntList paramIndexList_;
	static const Core::ParameterInt paramFrom_;
	static const Core::ParameterInt paramTo_;
	bool useIndexList_;
	std::vector<s32> indexList_;
	u32 from_;
	u32 to_;
public:
	ComponentSelectionPreprocessor(const char* name);
	virtual ~ComponentSelectionPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual bool needsContext() { return false; }
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

/*
 * random image cropping and flipping with different crop sizes at corners or center
 * as described in Want et.al.: Towards good practices for very deep two-stream convnets (2015)
 */
class RandomImageCroppingPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterInt paramInputWidth_;
	static const Core::ParameterInt paramInputHeight_;
	static const Core::ParameterInt paramChannels_;
	static const Core::ParameterIntList paramPossibleCropSideLengths_;
	static const Core::ParameterInt paramCropWidth_;
	static const Core::ParameterInt paramCropHeight_;
	u32 inputWidth_;
	u32 inputHeight_;
	u32 channels_;
	std::vector<u32> sideLengthList_;
	u32 cropWidth_;
	u32 cropHeight_;
	void createCrop(const Math::Matrix<Float>& in, Math::Matrix<Float>& out, u32 column);
public:
	RandomImageCroppingPreprocessor(const char* name);
	virtual ~RandomImageCroppingPreprocessor() {}
	virtual void initialize(u32 inputDimension);
	virtual bool needsContext() { return false; }
	virtual void work(const Math::Matrix<Float>& in, Math::Matrix<Float>& out);
};

} // namespace

#endif /* FEATURES_PREPROCESSOR_HH_ */
