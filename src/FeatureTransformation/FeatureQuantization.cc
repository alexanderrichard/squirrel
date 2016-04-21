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

#include "FeatureQuantization.hh"
#include <Features/FeatureWriter.hh>

using namespace FeatureTransformation;

/*
 * Quantizer
 */
// normalize over number of summed/quantized feature vectors
const Core::ParameterBool Quantizer::paramApplyHistogramNormalization_("apply-histogram-normalization", true, "feature-quantization");

const Core::ParameterBool Quantizer::paramApplyL2Normalization_("apply-l2-normalization", false, "feature-quantization");

const Core::ParameterBool Quantizer::paramApplyPowerNormalization_("apply-power-normalization", false, "feature-quantization");

const Core::ParameterFloat Quantizer::paramPowerNormalizationAlpha_("power-normalization-alpha", 0.5, "feature-quantization");

const Core::ParameterBool Quantizer::paramReduceDimension_("reduce-dimension", false, "feature-quantization");

const Core::ParameterString Quantizer::paramReductionMatrixFile_("reduction-matrix", "", "feature-quantization");

const Core::ParameterString Quantizer::paramReductionBiasFile_("reduction-bias", "", "feature-quantization");

Quantizer::Quantizer() :
		histogramNormalization_(Core::Configuration::config(paramApplyHistogramNormalization_)),
		powerNormalization_(Core::Configuration::config(paramApplyPowerNormalization_)),
		l2Normalization_(Core::Configuration::config(paramApplyL2Normalization_)),
		dimensionReduction_(Core::Configuration::config(paramReduceDimension_)),
		alpha_(Core::Configuration::config(paramPowerNormalizationAlpha_))
{}

void Quantizer::initialize() {
	if (histogramNormalization_) {
		Core::Log::os("Use histogram normalization in feature quantization.");
	}
	if (powerNormalization_) {
		Core::Log::os("Use power-normalization in feature quantization.");
	}
	if (l2Normalization_) {
		Core::Log::os("Use l2-normalization in feature quantization.");
	}
	if (dimensionReduction_) {
		// load weight matrix and bias vector
		require(!Core::Configuration::config(paramReductionMatrixFile_).empty());
		require(!Core::Configuration::config(paramReductionBiasFile_).empty());
		reductionMatrix_.read(Core::Configuration::config(paramReductionMatrixFile_));
		reductionBias_.read(Core::Configuration::config(paramReductionBiasFile_));
		reductionMatrix_.initComputation();
		reductionBias_.initComputation();
		Core::Log::os("Use dimension reduction in feature quantization.");
	}
	postprocessingResult_.initComputation();
}

void Quantizer::postprocessing(u32 nQuantizedFeatures) {
	/* histogram normalization */
	if (histogramNormalization_) {
		tmpHistogram_.scale(1.0 / nQuantizedFeatures);
	}
	/* power normalization */
	if (powerNormalization_) {
		tmpHistogram_.signedPow(alpha_);
	}
	/* l2 normalization */
	if (l2Normalization_) {
		Float l2Norm = tmpHistogram_.normEuclidean();
		if (l2Norm != 0) // if vector is all zero, leave it like that
			tmpHistogram_.scale(1.0 / l2Norm);
	}
	/* dimension reduction */
	postprocessingResult_.resize(outputDimension());
	if (dimensionReduction_) {
		postprocessingResult_.multiply(reductionMatrix_, tmpHistogram_, true);
		postprocessingResult_.add(reductionBias_);
	}
	else {
		postprocessingResult_.copy(tmpHistogram_);
	}
}

/*
 * FeatureQuantization
 */
const Core::ParameterInt FeatureQuantization::paramBatchSize_("batch-size", 1, "feature-quantization");

const Core::ParameterEnum FeatureQuantization::paramQuantizationType_("type", "none, bag-of-words, fisher-vector",
		"none", "feature-quantization");

FeatureQuantization::FeatureQuantization() :
		quantizer_(0),
		batchSize_(Core::Configuration::config(paramBatchSize_)),
		nObservations_(0),
		isInitialized_(false)
{}

void FeatureQuantization::initialize() {
	featureReader_.initialize();
	createQuantizer();
	quantizer_->initialize();
	isInitialized_ = true;
}

void FeatureQuantization::resetHistogram() {
	require(isInitialized_);
	quantizer_->tmpHistogram().setToZero();
	histogram_.setToZero();
	nObservations_ = 0;
}

void FeatureQuantization::createQuantizer() {
	switch ((QuantizationType) Core::Configuration::config(paramQuantizationType_)) {
	case bagOfWords:
		Core::Log::os("Create bag-of-words quantizer.");
		quantizer_ = new BagOfWordsQuantizer();
		break;
	case fisherVector:
		Core::Log::os("Create fisher-vector quantizer.");
		quantizer_ = new FisherVectorQuantizer();
		break;
	default:
		std::cerr << "FeatureQuantization: no quantization type selected. Abort." << std::endl;
		exit(1);
	}
}

void FeatureQuantization::generateBatch(const Math::Matrix<Float>& sequence, u32 from, u32 to) {
	require_le(from, to);
	require_lt(to, sequence.nColumns());
	require_le(to - from + 1, batchSize_);

	batch_.resize(sequence.nRows(), to - from + 1);
	batch_.finishComputation(false);
	for (u32 col = 0; col < to - from + 1; col++) {
		for (u32 row = 0; row < sequence.nRows(); row++) {
			batch_.at(row, col) = sequence.at(row, col + from);
		}
	}
	batch_.initComputation();
	nObservations_ += to - from + 1;
}

void FeatureQuantization::quantizeSequence(const Math::Matrix<Float>& sequence) {
	require(isInitialized_);

	resetHistogram();

	u32 from = 0;
	while (from < sequence.nColumns()) {
		u32 to = std::min(sequence.nColumns(), from + batchSize_) - 1;
		generateBatch(sequence, from, to);
		quantizer_->processBatch(batch_);
		from = to + 1;
	}

	// apply postprocessing
	quantizer_->postprocessing(nObservations_);

	// synchronize histogram
	quantizer_->postprocessingResult().finishComputation();
	for (u32 i = 0; i < quantizer_->postprocessingResult().nRows(); i++)
		histogram_.at(i) = quantizer_->postprocessingResult().at(i);
	quantizer_->postprocessingResult().initComputation(false);
}

void FeatureQuantization::quantizeFeatures() {
	Features::FeatureWriter featureWriter;
	histogram_.resize(quantizer_->outputDimension());
	while (featureReader_.hasSequences()) {
		quantizeSequence(featureReader_.next());
		// write result
		featureWriter.write(histogram_);
	}
	featureWriter.finalize();
}

/*
 * TemporalFeatureQuantization
 */
const Core::ParameterBool TemporalFeatureQuantization::paramStoreAsIntegralSequence_("store-as-integral-sequence", false,
		"feature-quantization");

const Core::ParameterBool TemporalFeatureQuantization::paramPrependMissingFrames_("prepend-missing-frames", true,
		"feature-quantization");

const Core::ParameterInt TemporalFeatureQuantization::paramEnsureSequenceLength_("ensure-sequence-length", 0,
		"feature-quantization");

const Core::ParameterInt TemporalFeatureQuantization::paramTemporalStride_("temporal-stride", 1,
		"feature-quantization");

TemporalFeatureQuantization::TemporalFeatureQuantization() :
		Precursor(),
		storeAsIntegralSequence_(Core::Configuration::config(paramStoreAsIntegralSequence_)),
		prependMissingFrames_(Core::Configuration::config(paramPrependMissingFrames_)),
		ensureSequenceLength_(Core::Configuration::config(paramEnsureSequenceLength_)),
		temporalStride_(Core::Configuration::config(paramTemporalStride_))
{
	require_ge(temporalStride_, 1);
}

void TemporalFeatureQuantization::initialize() {
	Precursor::initialize();
	Core::Log::openTag("temporal-sequence-quantization");
	Core::Log::os("Prepend missing frames: ") << (prependMissingFrames_ ? "true" : "false");
	Core::Log::os("Ensured sequence length (before temporal stride): ") << ensureSequenceLength_;
	Core::Log::os("Temporal stride: ") << temporalStride_;
	Core::Log::os("Store as integral sequence: ") << (storeAsIntegralSequence_ ? "true" : "false");
	Core::Log::closeTag();
}

void TemporalFeatureQuantization::quantizeSequence(const Math::Matrix<Float>& sequence, std::vector<u32>& timestamps) {
	require(isInitialized_);
	require_le(timestamps.at(0), timestamps.back());

	histogram_.resize(quantizer_->outputDimension(), (timestamps.back() - timestamps.at(0)) / temporalStride_ + 1);
	resetHistogram();

	u32 from = 0;
	while (from < sequence.nColumns()) {
		u32 t = (timestamps.at(from) - timestamps.at(0)) / temporalStride_;
		// determine part of sequence labeled with current timestamp t (divided by the temporal stride)
		u32 to = from;
		while ((to < sequence.nColumns()) && (timestamps.at(from) / temporalStride_ == timestamps.at(to) / temporalStride_)) {
			to++;
		}
		to--;
		// process all observations for this timestamp/temporal stride
		while (nObservations_ <= to) {
			u32 batchSize = std::min(to - nObservations_ + 1, batchSize_);
			generateBatch(sequence, nObservations_, nObservations_ + batchSize - 1);
			quantizer_->processBatch(batch_);
		}
		// apply post-processing
		quantizer_->postprocessing(to - from + 1);

		from = nObservations_;
		// synchronize histogram
		quantizer_->postprocessingResult().finishComputation();
		for (u32 i = 0; i < quantizer_->postprocessingResult().nRows(); i++) {
			histogram_.at(i, t) += quantizer_->postprocessingResult().at(i);
		}
		quantizer_->postprocessingResult().initComputation(false);
		quantizer_->postprocessingResult().setToZero();
		quantizer_->tmpHistogram().setToZero();
	}
}

void TemporalFeatureQuantization::resetHistogram() {
	require(isInitialized_);
	quantizer_->tmpHistogram().setToZero();
	histogram_.setToZero();
	nObservations_ = 0;
}

void TemporalFeatureQuantization::quantizeFeatures() {
	Features::SequenceFeatureWriter featureWriter;
	std::vector<u32> timestamps;
	Math::Matrix<Float> result;
	while (featureReader_.hasSequences()) {
		const Math::Matrix<Float>& sequence = featureReader_.next();
		quantizeSequence(sequence, featureReader_.currentTimestamps());
		// compute final result sequence
		u32 nFrames = histogram_.nColumns();
		if (prependMissingFrames_)
			nFrames += featureReader_.currentTimestamps().at(0) / temporalStride_;
		nFrames = std::max(nFrames, ensureSequenceLength_ / temporalStride_);
		result.resize(histogram_.nRows(), nFrames);
		result.setToZero();
		u32 startCol = (prependMissingFrames_ ? featureReader_.currentTimestamps().at(0) / temporalStride_ : 0);
		result.copyBlockFromMatrix(histogram_, 0, 0, 0, startCol, histogram_.nRows(), histogram_.nColumns());
		// compute integral sequence if required
		if (storeAsIntegralSequence_) {
			for (u32 t = 1; t < result.nColumns(); t++) {
#pragma omp parallel for
				for (u32 d = 0; d < result.nRows(); d++) {
					result.at(d, t) += result.at(d, t-1);
				}
			}
		}
		// generate timestamps
		timestamps.resize(nFrames);
		for (u32 i = 0; i < nFrames; i++) {
			timestamps.at(i) = i;
		}
		// write result
		featureWriter.write(timestamps, result);
	}
	featureWriter.finalize();
}

/*
 * BagOfWordsQuantizer
 */
const Core::ParameterString BagOfWordsQuantizer::paramVisualVocabulary_("visual-vocabulary", "",
		"feature-quantization.bag-of-words");

const Core::ParameterString BagOfWordsQuantizer::paramLambdaFile_("lambda-file", "",
		"feature-quantization.bag-of-words");

const Core::ParameterString BagOfWordsQuantizer::paramBiasFile_("bias-file", "",
		"feature-quantization.bag-of-words");

const Core::ParameterBool BagOfWordsQuantizer::paramUseMaximumApproximation_("use-maximum-approximation", true,
		"feature-quantization.bag-of-words");

BagOfWordsQuantizer::BagOfWordsQuantizer() :
		Precursor(),
		visualVocabularyFile_(Core::Configuration::config(paramVisualVocabulary_)),
		lambdaFile_(Core::Configuration::config(paramLambdaFile_)),
		biasFile_(Core::Configuration::config(paramBiasFile_)),
		maximumApproximation_(Core::Configuration::config(paramUseMaximumApproximation_)),
		isInitialized_(false)
{}

void BagOfWordsQuantizer::readParameters() {
	if (visualVocabularyFile_.empty()) {
		require(!lambdaFile_.empty());
		require(!biasFile_.empty());
		alpha_.read(biasFile_);
		lambda_.read(lambdaFile_);
	}
	else {
		Math::Matrix<Float> means;
		means.read(visualVocabularyFile_);
		// convert to model with lambda and alpha
		alpha_.resize(means.nRows());
		alpha_.setToZero();
		lambda_.resize(means.nColumns(), means.nRows());
		for (u32 c = 0; c < means.nRows(); c++) {
			for (u32 f = 0; f < means.nColumns(); f++)
			{
				lambda_.at(f, c) = means.at(c, f);
				alpha_.at(c) -=  0.5 * means.at(c,f) * means.at(c,f);
			}
		}
	}
}

void BagOfWordsQuantizer::initialize() {
	Precursor::initialize();
	readParameters();
	lambda_.initComputation();
	alpha_.initComputation();
	dist_.initComputation();
	tmpHistogram_.initComputation();
	tmpHistogram_.resize(alpha_.size());
	isInitialized_ = true;
}

u32 BagOfWordsQuantizer::outputDimension() {
	require(isInitialized_);
	if (dimensionReduction_)
		return reductionMatrix_.nColumns();
	else
		return alpha_.size();
}

void BagOfWordsQuantizer::processBatch(Math::CudaMatrix<Float>& batch) {
	require(batch.isComputing());
	dist_.resize(alpha_.size(), batch.nColumns());
	dist_.setToZero();
	dist_.addMatrixProduct(lambda_, batch, (Float)1.0, (Float)1.0, true, false);
	dist_.addToAllColumns(alpha_);
	if (maximumApproximation_)
		dist_.max();
	else
		dist_.softmax();
	tmpHistogram_.addSummedColumns(dist_);
}

/*
 * FisherVectorQuantizer
 */
const Core::ParameterString FisherVectorQuantizer::paramMeanFile_("mean-file", "",
		"feature-quantization.fisher-vector");

const Core::ParameterString FisherVectorQuantizer::paramVarianceFile_("variance-file", "",
		"feature-quantization.fisher-vector");

const Core::ParameterString FisherVectorQuantizer::paramWeightsFile_("weights-file", "",
		"feature-quantization.fisher-vector");

const Core::ParameterBool FisherVectorQuantizer::paramSecondOrderFisherVectors_("second-order-fisher-vectors", true,
		"feature-quantization.fisher-vector");

const Core::ParameterBool FisherVectorQuantizer::paramUseMaximumApproximation_("use-maximum-approximation", false,
		"feature-quantization.fisher-vector");

FisherVectorQuantizer::FisherVectorQuantizer() :
		Precursor(),
		meanFile_(Core::Configuration::config(paramMeanFile_)),
		varianceFile_(Core::Configuration::config(paramVarianceFile_)),
		weightsFile_(Core::Configuration::config(paramWeightsFile_)),
		dimension_(0),
		nMixtures_(0),
		secondOrderVectors_(Core::Configuration::config(paramSecondOrderFisherVectors_)),
		maximumApproximation_(Core::Configuration::config(paramUseMaximumApproximation_)),
		isInitialized_(false)
{}

void FisherVectorQuantizer::readParameters() {
	require(!meanFile_.empty());
	require(!varianceFile_.empty());
	require(!weightsFile_.empty());
	Math::Matrix<Float> means;
	Math::Matrix<Float> variances;
	Math::Vector<Float> weights;
	means.read(meanFile_, true);
	variances.read(varianceFile_, true);
	weights.read(weightsFile_);
	dimension_ = means.nRows();
	nMixtures_ = means.nColumns();
	require_eq(means.nRows(), variances.nRows());
	require_eq(means.nColumns(), variances.nColumns());
	lambda_.resize(dimension_ * 2, nMixtures_);
	bias_.resize(nMixtures_);
	// fill with zero
	for (u32 i = 0; i < bias_.size(); i++) {
		bias_.at(i) = 0;
	}
	means_.resize(dimension_ * nMixtures_);
	stdDev_.resize(dimension_ * nMixtures_);
	sqrtWeights_.resize(nMixtures_);
	for (u32 k = 0; k < nMixtures_; k++) {
		for (u32 d = 0; d < dimension_; d++) {
			lambda_.at(d, k) = means.at(d, k) / variances.at(d, k);
			lambda_.at(d + dimension_, k) = -0.5 / variances.at(d, k);
			bias_.at(k) += pow(means.at(d, k), 2) / variances.at(d, k) + log(variances.at(d, k));
			means_.at(k * dimension_ + d) = means.at(d, k);
			stdDev_.at(k * dimension_ + d) = sqrt(variances.at(d, k));
		}
		bias_.at(k) = log(weights.at(k)) -0.5 * (bias_.at(k) + dimension_ * log(2*M_PI));
		sqrtWeights_.at(k) = sqrt(weights.at(k));
	}
}

void FisherVectorQuantizer::initialize() {
	Precursor::initialize();
	readParameters();
	sqrtWeights_.initComputation();
	means_.initComputation();
	stdDev_.initComputation();
	lambda_.initComputation();
	bias_.initComputation();
	gamma_.initComputation();
	tmpMatrix_.initComputation();
	tmpVector_.initComputation();
	tmpHistogram_.initComputation();
	tmpHistogram_.resize(dimension_ * nMixtures_ * (secondOrderVectors_ ? 2 : 1));
	isInitialized_ = true;
}

void FisherVectorQuantizer::computeGamma(Math::CudaMatrix<Float>& batch) {
	tmpMatrix_.resize(dimension_ * 2, batch.nColumns());
	tmpMatrix_.setToDiagonalSecondOrderFeatures(batch);
	gamma_.resize(nMixtures_, tmpMatrix_.nColumns());
	gamma_.setToZero();
	gamma_.addMatrixProduct(lambda_, tmpMatrix_, 0, 1, true, false);
	gamma_.addToAllColumns(bias_);
	if (maximumApproximation_)
		gamma_.max();
	else
		gamma_.softmax();
}

u32 FisherVectorQuantizer::outputDimension() {
	require(isInitialized_);
	if (dimensionReduction_)
		return reductionMatrix_.nColumns();
	else
		return dimension_ * nMixtures_ * (secondOrderVectors_ ? 2 : 1);
}

void FisherVectorQuantizer::processBatch(Math::CudaMatrix<Float>& batch) {
	require(isInitialized_);
	computeGamma(batch);
	// divide gamma by sqrtWeights_
	gamma_.divideRowsByScalars(sqrtWeights_);
	// expand batch
	tmpMatrix_.resize(dimension_ * nMixtures_, batch.nColumns());
	tmpMatrix_.clone(batch, nMixtures_);
	// subtract means and divide by standard deviation
	tmpMatrix_.addToAllColumns(means_, -1);
	tmpMatrix_.divideRowsByScalars(stdDev_);
	batch.swap(tmpMatrix_);
	// expand gamma
	tmpMatrix_.resize(dimension_ * nMixtures_, batch.nColumns());
	tmpMatrix_.cloneElementwise(gamma_, dimension_);
	gamma_.copyStructure(tmpMatrix_);
	gamma_.copy(tmpMatrix_); // expanded gamma is now stored in tmpMatrix_ and gamma_
	// first order using gamma_
	gamma_.elementwiseMultiplication(batch);
	// second order using tmpMatrix_
	if (secondOrderVectors_) {
		batch.elementwiseMultiplication(batch);
		batch.addConstantElementwise(-1.0);
		tmpMatrix_.elementwiseMultiplication(batch);
		// divide by sqrt(2)
		tmpMatrix_.scale(1.0 / sqrt(2));
	}
	// accumulate results and copy to tmpHistogram
	tmpVector_.resize(dimension_ * nMixtures_);
	tmpVector_.copyBlockFromVector(tmpHistogram_, 0, 0, dimension_ * nMixtures_);
	tmpVector_.addSummedColumns(gamma_);
	tmpHistogram_.copyBlockFromVector(tmpVector_, 0, 0, dimension_ * nMixtures_);
	if (secondOrderVectors_) {
		tmpVector_.copyBlockFromVector(tmpHistogram_, dimension_ * nMixtures_, 0, dimension_ * nMixtures_);
		tmpVector_.addSummedColumns(tmpMatrix_);
		tmpHistogram_.copyBlockFromVector(tmpVector_, 0, dimension_ * nMixtures_, dimension_ * nMixtures_);
	}
}
