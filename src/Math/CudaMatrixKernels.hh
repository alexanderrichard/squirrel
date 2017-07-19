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

#ifndef MATH_CUDAKERNELS_HH_
#define MATH_CUDAKERNELS_HH_

// mixed precision methods
void _cuda_axpy(int n, float alpha, const float *x, double *y);

// own kernels

// sign(x) * pow(|x|,p)
template<typename T>
void _cuda_signedPow(T *data, unsigned int nRows, unsigned int nColumns, T p);

// exp
template<typename T>
void _cuda_exp(T *data, unsigned int nRows, unsigned int nColumns);

// log
template<typename T>
void _cuda_log(T *data, unsigned int nRows, unsigned int nColumns);

// sin
template<typename T>
void _cuda_sin(T *data, unsigned int nRows, unsigned int nColumns);

// cos
template<typename T>
void _cuda_cos(T *data, unsigned int nRows, unsigned int nColumns);

// asin
template<typename T>
void _cuda_asin(T *data, unsigned int nRows, unsigned int nColumns);

// acos
template<typename T>
void _cuda_acos(T *data, unsigned int nRows, unsigned int nColumns);

// abs
template<typename T>
void _cuda_abs(T *data, unsigned int nRows, unsigned int nColumns);

// tanh
template<typename T>
void _cuda_tanh(T *data, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_sigmoid(T gamma, T *data, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_triangle(T *data, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_sum(T *data, unsigned int nRows, unsigned int nColumns, T *result);

template<typename T>
void _cuda_columnwiseSquaredEuclideanDistance(const T *A, unsigned int nRows, unsigned int nColumns, const T *v, T *result);

template<typename T>
void _cuda_clone(const T *dataA, T *dataB, unsigned int nRowsB, unsigned int nColumnsB, unsigned int nClones);

template<typename T>
void _cuda_cloneElementwise(const T *dataA, T *dataB, unsigned int nRowsB, unsigned int nColumnsB, unsigned int nClones);

template<typename T>
void _cuda_addElementsByModuloIndex(const T *dataA, T *dataB, unsigned int nRowsA, unsigned int nRowsB, unsigned int nColumns);

template<typename T>
void _cuda_chiSquareFeatureMap(const T *dataA, T *dataB, unsigned int nElementsB, unsigned int n, T samplingDistance, T min);

template<typename T>
void _cuda_histogramIntersectionFeatureMap(const T *dataA, T *dataB, unsigned int nElementsB, unsigned int n, T samplingDistance, T min);

template<typename T>
void _cuda_elementwiseMultiplicationWithApproximateFeatureMapDerivative(const T *dataA, T *dataB, unsigned int nElements, unsigned int n, T samplingDistance, T kappa0);

template<typename T>
void _cuda_addSummedRows(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale);

template<typename T>
void _cuda_addSummedRows(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, T *tmpDevPtr, unsigned int tmpRows, const T scale);

template<typename T>
void _cuda_addSummedColumns(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale);

template<typename T>
void _cuda_addSummedColumnsChannelWise(T *vector, const T* matrix, const unsigned int channels, const unsigned int nRows, const unsigned int nColumns, const T scale);

// square matrix elementwise and add sum of columns to vector
template<typename T>
void _cuda_addSquaredSummedColumns(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale);

template<typename T>
void _cuda_addSummedNeighborsInARow(T* dataA, const T* dataB, unsigned int rowsA, unsigned int columnsA, unsigned int nNeighbors);

template<typename T>
void _cuda_addWeighted(T* res, const T *X, const T *weights, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_elementwiseMultiplication(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_elementwiseDivision(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_rpropUpdate(T *currentValues, T *newGradients, T *oldGradients, T *updateValues, T increasingFactor, T decreasingFactor,
		T maxUpdateValue, T minUpdateValue, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_addConstantElementwise(T constant, T *data, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_getMaxOfColumns(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_getMaxOfColumns(T *vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, T *tmpDevPtr, unsigned int tmpRows);

template<typename T>
void _cuda_elementwiseMultiplicationWithSigmoidDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_elementwiseMultiplicationWithTriangleDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_elementwiseMultiplicationWithTanhDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_multiplicationWithSoftmaxDerivative(T *data, T *datab, T *datac, unsigned int nRows, unsigned int nColumns);

template <typename T>
void _cuda_elementwiseMultiplicationWithClippedDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns, T thresholdLeft, T thresholdRight);

template <typename T>
void _cuda_elementwiseMultiplicationWithLogDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns);

template <typename T>
void _cuda_elementwiseMultiplicationWithSignedPowDerivative(T *data, T *datab, unsigned int nRows, unsigned int nColumns, T p);

template<typename T>
void _cuda_multiplicationWithL2NormalizationDerivative(T *data, T *datab, T *datac, T* datad, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_addToAllColumns(T *data, T *datab, unsigned int nRows, unsigned int nColumns, T alpha);

template<typename T>
void _cuda_addToAllChannels(T *a, T *b, unsigned int channels, unsigned int nRows, unsigned int nColumns, T alpha);

template<typename T>
void _cuda_addToAllRows(T *data, T *datab, unsigned int nRows, unsigned int nColumns, T alpha);

template<typename T>
void _cuda_multiplyColumnsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_divideColumnsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_multiplyRowsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_divideRowsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_avgPool(const T *source, T *result, const unsigned int sourceRows,
	const unsigned int sourceColumns, const unsigned int sourceWidth, const unsigned int sourceHeight,
	const unsigned int sourceChannels, const unsigned int poolSize, const unsigned int stride);
template<typename T>
void _cuda_backPropogateAvgPool( T *result, const T *errorSignalOut, const unsigned int sourceRows,
		const unsigned int sourceColumns, const unsigned int sourceWidth, const unsigned int sourceHeight,
		const unsigned int sourceChannels, const unsigned int poolSize,	const unsigned int stride);
template<typename T>
void _cuda_maxPool(const T *source, T *result, const unsigned int sourceRows,
	const unsigned int sourceColumns, const unsigned int sourceWidth, const unsigned int sourceHeight,
	const unsigned int sourceChannels, const unsigned int poolSize, const unsigned int stride);
template<typename T>
void _cuda_backPropogateMaxPool(T *result, const T* activationIn, const T* activationOut, const T *errorSignal,
		const unsigned int sourceRows, const unsigned int sourceColumns, const unsigned int sourceWidth,
		const unsigned int sourceHeight, const unsigned int sourceChannels, const unsigned int poolSize,
		const unsigned int stride);

template<typename T>
void _cuda_prepareConvolution(T* dest, const T* source, const unsigned int sourceWidth, const unsigned int sourceHeight,
		const unsigned int sourceChannels, const unsigned int kernelWidth, const unsigned int kernelHeight,
		const unsigned int destRows, const unsigned int destCols, const unsigned int strideX, const unsigned int strideY);
template<typename T>
void _cuda_prepareConvolutionBackProp(T* dest, const T* source, const unsigned int destWidth, const unsigned int destHeight,
		const unsigned int destChannels, const unsigned int kernelWidth, const unsigned int kernelHeight, const unsigned int destRows, const unsigned int destCols);
template<typename T>
void _cuda_prepareConvolutionSame(T* dest, const T* source, const unsigned int sourceWidth, const unsigned int sourceHeight,
		const unsigned int sourceChannels, const unsigned int kernelWidth, const unsigned int kernelHeight,
		const unsigned int destRows, const unsigned int destCols, const unsigned int strideX, const unsigned int strideY);
template<typename T>
void _cuda_prepareConvolutionSameBackProp(T* dest, const T* source, const unsigned int destWidth, const unsigned int destHeight,
		const unsigned int destChannels, const unsigned int kernelWidth, const unsigned int kernelHeight,
		const unsigned int destRows, const unsigned int destCols, const unsigned int strideX, const unsigned int strideY);
template<typename T>
void _cuda_rearrange(T *dest, const T *source, const unsigned int sourceRows,
		const unsigned int destRows, const unsigned int destColumns, const unsigned int destNumPixels);
template<typename T>
void _cuda_rearrangeBackProp(T *dest, const T *source, const unsigned int sourceColumns,
		const unsigned int destRows, const unsigned int destColumns, const unsigned int destNumPixels);

template<typename T>
void _cuda_fill(T *devPtr, T value, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_ensureMinimalValue(T *devPtr, T value, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_ensureMaximalValue(T *devPtr, T value, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_argMax(T *devPtr, unsigned int nRows, unsigned int nColumns, unsigned int *resultDev);

template<typename T>
void _cuda_max(T *devResult, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_max(T *devResult, const T *devA, const T *devB, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_elementwiseMultiplicationWithKroneckerDelta(T *devResult, const T *devA, const T *devB, unsigned int nRows, unsigned int nColumns);

template<typename T>
void _cuda_nClassificationErrors(T *devPtr, unsigned int nRows, unsigned int nColumns, T *targets, unsigned int *resultDev);

template<typename T>
void _cuda_crossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev);

template<typename T>
void _cuda_weightedCrossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev, T *weights);

template<typename T>
void _cuda_squaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev);

template<typename T>
void _cuda_weightedSquaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev, T *weights);

template<typename T>
void _cuda_smoothedL1ObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev);

template<typename T>
void _cuda_weightedSmoothedL1ObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T* weights, T *resultDev);

template<typename T>
void _cuda_appendSecondOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset);

template<typename T>
void _cuda_appendDiagonalSecondOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset);

template<typename T>
void _cuda_appendThirdOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset);

template<typename T>
void _cuda_appendDiagonalThirdOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset);

template<typename T>
void _cuda_gaussianMixturePosteriors(T *P, const T *X, const T *means, const T *variances, const T *weights, unsigned int nFeatures, unsigned int featureDim, unsigned int nMixtures);

template<typename T>
void _cuda_fisherEncoding(T *F, const T *X, const T *means, const T *variances, const T *weights, const T* gamma, unsigned int nFeatures, unsigned int featureDim, unsigned int nMixtures);

template<typename T>
void _cuda_dropout(T *X, const T *mask, unsigned int nRows, unsigned int nColumns, T dropoutProbability);

#endif /* MATH_CUDAKERNELS_HH_ */
