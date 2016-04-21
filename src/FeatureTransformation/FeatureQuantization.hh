#ifndef FEATURETRANSFORMATION_FEATUREQUANTIZATION_HH_
#define FEATURETRANSFORMATION_FEATUREQUANTIZATION_HH_

#include "Features/FeatureReader.hh"
#include "Math/CudaMatrix.hh"
#include "Math/CudaVector.hh"

namespace FeatureTransformation {

/*
 * base class for a quantization algorithm
 */
class Quantizer
{
private:
	static const Core::ParameterBool paramApplyHistogramNormalization_;
	static const Core::ParameterBool paramApplyPowerNormalization_;
	static const Core::ParameterBool paramApplyL2Normalization_;
	static const Core::ParameterFloat paramPowerNormalizationAlpha_;
	static const Core::ParameterBool paramReduceDimension_;
	static const Core::ParameterString paramReductionMatrixFile_;
	static const Core::ParameterString paramReductionBiasFile_;
protected:
	Math::CudaVector<Float> tmpHistogram_;
	bool histogramNormalization_;
	bool powerNormalization_;
	bool l2Normalization_;
	bool dimensionReduction_;
	Math::CudaMatrix<Float> reductionMatrix_;
	Math::CudaVector<Float> reductionBias_;
	Math::CudaVector<Float> postprocessingResult_;
	Float alpha_;
public:
	Quantizer();
	virtual ~Quantizer() {}
	virtual void initialize();
	/*
	 * quantize all features that are collected in batch
	 * @param batch the bunch of features to be quantized in parallel
	 */
	virtual void processBatch(Math::CudaMatrix<Float>& batch) = 0;
	/*
	 * postprocessing, e.g. l2-normalization
	 */
	virtual void postprocessing(u32 nQuantizedFeatures);
	virtual u32 outputDimension() = 0;
	Math::CudaVector<Float>& tmpHistogram() { return tmpHistogram_; }
	Math::CudaVector<Float>& postprocessingResult() { return postprocessingResult_; }
	/*
	 * return whether these options are active or not
	 */
	virtual bool applyHistogramNormalization() { return histogramNormalization_; }
	virtual bool applyPowerNormalization() { return powerNormalization_; }
	virtual bool applyL2Normalization() { return l2Normalization_; }
	virtual bool applyDimensionReduction() { return dimensionReduction_; }
};

/*
 * quantization of sequences of feature vectors
 */
class FeatureQuantization {
private:
	static const Core::ParameterInt paramBatchSize_;
	static const Core::ParameterEnum paramQuantizationType_;
	enum QuantizationType { none, bagOfWords, fisherVector };
protected:
	Math::CudaMatrix<Float> batch_;
	Math::Vector<Float> histogram_;
	Features::SequenceFeatureReader featureReader_;
	Quantizer* quantizer_;
	u32 batchSize_;
	u32 nObservations_;
	bool isInitialized_;

	virtual void resetHistogram();
	void createQuantizer();
	void generateBatch(const Math::Matrix<Float>& sequence, u32 from, u32 to);
	void quantizeSequence(const Math::Matrix<Float>& sequence);
public:
	FeatureQuantization();
	virtual ~FeatureQuantization() {}
	virtual void initialize();
	void quantizeFeatures();
};

/*
 * quantization on timeframe level
 */
class TemporalFeatureQuantization : public FeatureQuantization
{
private:
	static const Core::ParameterBool paramStoreAsIntegralSequence_;
	static const Core::ParameterBool paramPrependMissingFrames_;
	static const Core::ParameterInt paramEnsureSequenceLength_;
	static const Core::ParameterInt paramTemporalStride_;
protected:
	typedef FeatureQuantization Precursor;

	Math::Matrix<Float> histogram_;
	Math::CudaVector<Float> tmpVector_; // for integral sequences
	bool storeAsIntegralSequence_;
	bool prependMissingFrames_;
	u32 ensureSequenceLength_;
	u32 temporalStride_;

	virtual void resetHistogram();
	void quantizeSequence(const Math::Matrix<Float>& sequence, std::vector<u32>& timestamps);
public:
	TemporalFeatureQuantization();
	~TemporalFeatureQuantization() {}
	virtual void initialize();
	void quantizeFeatures();
};

/*
 * Bag-of-Words quantizer
 */
class BagOfWordsQuantizer : public Quantizer
{
private:
	typedef Quantizer Precursor;
	// the means
	static const Core::ParameterString paramVisualVocabulary_;
	// can be given instead of the means
	static const Core::ParameterString paramLambdaFile_;
	static const Core::ParameterString paramBiasFile_;
	static const Core::ParameterBool paramUseMaximumApproximation_;
protected:
	std::string visualVocabularyFile_;
	std::string lambdaFile_;
	std::string biasFile_;
	bool maximumApproximation_;
	bool isInitialized_;
	// nearest mean can be computed as \Lambda x + \alpha, therefore matrix lambda and vector alpha
	Math::CudaMatrix<Float> lambda_;
	Math::CudaVector<Float> alpha_;
	Math::CudaMatrix<Float> dist_;
protected:
	virtual void readParameters();
public:
	BagOfWordsQuantizer();
	virtual ~BagOfWordsQuantizer() {}
	void initialize();
	virtual void processBatch(Math::CudaMatrix<Float>& batch);
	virtual u32 outputDimension();
};

/*
 * Fisher-vector quantizer
 */
class FisherVectorQuantizer : public Quantizer
{
private:
	typedef Quantizer Precursor;
	// Gaussian mixture model parameters
	static const Core::ParameterString paramMeanFile_;
	static const Core::ParameterString paramVarianceFile_;
	static const Core::ParameterString paramWeightsFile_;
	static const Core::ParameterBool paramSecondOrderFisherVectors_;
	static const Core::ParameterBool paramUseMaximumApproximation_;
protected:
	std::string meanFile_;
	std::string varianceFile_;
	std::string weightsFile_;
	u32 dimension_;
	u32 nMixtures_;
	bool secondOrderVectors_;
	bool maximumApproximation_;
	bool isInitialized_;
	Math::CudaMatrix<Float> lambda_;
	Math::CudaVector<Float> bias_;
	Math::CudaVector<Float> means_;
	Math::CudaVector<Float> stdDev_;
	Math::CudaVector<Float> sqrtWeights_;
	Math::CudaMatrix<Float> gamma_;
	Math::CudaMatrix<Float> tmpMatrix_;
	Math::CudaVector<Float> tmpVector_;
protected:
	virtual void readParameters();
	void computeGamma(Math::CudaMatrix<Float>& batch);
public:
	FisherVectorQuantizer();
	virtual ~FisherVectorQuantizer() {}
	void initialize();
	virtual void processBatch(Math::CudaMatrix<Float>& batch);
	virtual u32 outputDimension();
};

} // namespace

#endif /* FEATURETRANSFORMATION_FEATUREQUANTIZATION_HH_ */
