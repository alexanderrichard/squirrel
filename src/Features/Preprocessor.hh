#ifndef FEATURES_PREPROCESSOR_HH_
#define FEATURES_PREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

namespace Features {

/*
 * Base class for feature preprocessors
 * preprocessors are applied when the features are read from the cache file directly
 */
class Preprocessor
{
private:
	static const Core::ParameterEnum paramType_;
	enum Type { none, vectorSubtraction, vectorDivision, matrixMultiplication, featureSelection, windowing, zscore, componentSelection };
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
 * feature selection
 */
class FeatureSelectionPreprocessor : public Preprocessor
{
private:
	typedef Preprocessor Precursor;
	static const Core::ParameterInt paramStartIndex_;
	static const Core::ParameterInt paramEndIndex_;
protected:
	u32 startIndex_;
	u32 endIndex_;
public:
	FeatureSelectionPreprocessor(const char* name);
	virtual ~FeatureSelectionPreprocessor() {}
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

} // namespace

#endif /* FEATURES_PREPROCESSOR_HH_ */
