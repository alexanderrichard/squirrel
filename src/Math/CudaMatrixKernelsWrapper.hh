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

#ifndef MATH_CUDAMATRIXKERNELSWRAPPER_HH_
#define MATH_CUDAMATRIXKERNELSWRAPPER_HH_

#include <Modules.hh>
#include <stdlib.h>
#include <iostream>

/**
 * Macro CUDACALL inserts the first parameter, if MODULE_CUDA is enabled.
 * Otherwise, a critical error is raised.
 */
#ifdef MODULE_CUDA
#include "CudaMatrixKernels.hh"
#define CUDACALL(function, functionName) \
		function;
#else
#define CUDACALL(function, functionName) \
		std::cerr << "Calling CUDA kernel " << functionName << " in a binary without GPU support!" << std::endl; \
		exit(1);
#endif

namespace Math {

namespace Cuda {

// exponential function

template <typename T>
inline void exp(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_exp<T>(devPtr, nRows, nColumns)), "exp");
}

// power function on absolute values, sign of original value is kept

template <typename T>
inline void signedPow(T *devPtr, unsigned int nRows, unsigned int nColumns, T p){
	CUDACALL((_cuda_signedPow<T>(devPtr, nRows, nColumns, p)), "signedPow");
}

// logarithm

template <typename T>
inline void log(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_log<T>(devPtr, nRows, nColumns)), "log");
}

// sine

template <typename T>
inline void sin(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_sin<T>(devPtr, nRows, nColumns)), "sin");
}

// cosine

template <typename T>
inline void cos(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_cos<T>(devPtr, nRows, nColumns)), "cos");
}

// arc sine

template <typename T>
inline void asin(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_asin<T>(devPtr, nRows, nColumns)), "asin");
}

// arc cosine

template <typename T>
inline void acos(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_acos<T>(devPtr, nRows, nColumns)), "acos");
}

// absolute value

template <typename T>
inline void abs(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_abs<T>(devPtr, nRows, nColumns)), "abs");
}

// addSummedRows

template <typename T>
inline void addSummedRows(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale){
	CUDACALL((_cuda_addSummedRows<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
			"addSummedRows");
}

// faster version of addSummedRows

template <typename T>
inline void addSummedRows(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns,
		T *tmpDevPtr, unsigned int tmpRows, const T scale){
	CUDACALL((_cuda_addSummedRows<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns,tmpDevPtr, tmpRows, scale)),
			"addSummedRows");
}

// addSummedColumns

template <typename T>
inline void addSummedColumns(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale){
	CUDACALL((_cuda_addSummedColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
			"addSummedColumns");
}

// addSummedColumnsChannelWise
template <typename T>
inline void addSummedColumnsChannelWise(T *vector, const T* matrix, const unsigned int channels, const unsigned int nRows, const unsigned int nColumns, const T scale) {
	CUDACALL((_cuda_addSummedColumnsChannelWise(vector, matrix, channels, nRows, nColumns, scale)),
			"addSummedColumnsChannelWise");
}

// addSquaredSummedColumns

template <typename T>
inline void addSquaredSummedColumns(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale){
	CUDACALL((_cuda_addSquaredSummedColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
			"addSquaredSummedColumns");
}

// addSummedNeighborsInARow

template <typename T>
inline void addSummedNeighborsInARow(T* dataA, const T* dataB, unsigned int rowsA, unsigned int columnsA, unsigned int nNeighbors){
	CUDACALL((_cuda_addSummedNeighborsInARow<T>(dataA, dataB, rowsA, columnsA, nNeighbors)),
			"addSummedNeighborsInARow");
}

// addWeighted

template <typename T>
inline void addWeighted(T* res, const T *X, const T *weights, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_addWeighted<T>(res, X, weights, nRows, nColumns)),
			"addWeighted");
}

// tanh

template <typename T>
inline void tanh(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_tanh<T>(devPtr, nRows, nColumns)), "tanh");
}

// sigmoid

template<typename T>
inline void sigmoid(T gamma, T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_sigmoid<T>(gamma, devPtr, nRows, nColumns)), "sigmoid");
}

// triangle

template<typename T>
inline void triangle(T *devPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_triangle<T>(devPtr, nRows, nColumns)), "triangle");
}

// sum

template<typename T>
inline void sum(T *devPtr, unsigned int nRows, unsigned int nColumns, T *resultDev){
	CUDACALL((_cuda_sum<T>(devPtr, nRows, nColumns, resultDev)), "sum");
}

// columnwiseSquaredEuclideanDistance

template<typename T>
inline void columnwiseSquaredEuclideanDistance(const T *A, unsigned int nRows, unsigned int nColumns, const T *v, T *result){
	CUDACALL((_cuda_columnwiseSquaredEuclideanDistance<T>(A, nRows, nColumns, v, result)), "columnwiseSquaredEuclideanDistance");
}

// clone

template<typename T>
inline void clone(const T *dataA, T *dataB, unsigned int nRowsB, unsigned int nColumnsB, unsigned int nClones){
	CUDACALL((_cuda_clone<T>(dataA, dataB, nRowsB, nColumnsB, nClones)),
			"clone");
}

// cloneElementwise

template<typename T>
inline void cloneElementwise(const T *dataA, T *dataB, unsigned int nRowsB, unsigned int nColumnsB, unsigned int nClones){
	CUDACALL((_cuda_cloneElementwise<T>(dataA, dataB, nRowsB, nColumnsB, nClones)),
			"cloneElementwise");
}

// addElementsByModuloIndex

template<typename T>
inline void addElementsByModuloIndex(const T *dataA, T *dataB, unsigned int nRowsA, unsigned int nRowsB, unsigned int nColumns){
	CUDACALL((_cuda_addElementsByModuloIndex<T>(dataA, dataB, nRowsA, nRowsB, nColumns)),
			"addElementsByModuloIndex");
}

// chi-square feature map

template<typename T>
inline void chiSquareFeatureMap(const T *dataA, T *dataB, unsigned int nElementsB, unsigned int n, T samplingDistance, T min){
	CUDACALL((_cuda_chiSquareFeatureMap<T>(dataA, dataB, nElementsB, n, samplingDistance, min)),
			"chiSquareFeatureMap");
}

// histogram intersection feature map

template<typename T>
inline void histogramIntersectionFeatureMap(const T *dataA, T *dataB, unsigned int nElementsB, unsigned int n, T samplingDistance, T min){
	CUDACALL((_cuda_histogramIntersectionFeatureMap<T>(dataA, dataB, nElementsB, n, samplingDistance, min)),
			"histogramIntersectionFeatureMap");
}

// multiplication with chi-square feature map derivative

template<typename T>
inline void elementwiseMultiplicationWithApproximateFeatureMapDerivative(const T *dataA, T *dataB, unsigned int nElements, unsigned int n, T samplingDistance, T kappa0){
	CUDACALL((_cuda_elementwiseMultiplicationWithApproximateFeatureMapDerivative<T>(dataA, dataB, nElements, n, samplingDistance, kappa0)),
			"elementwiseMultiplicationWithApproximateFeatureMapDerivative");
}

// elementwise multiplication

template<typename T>
inline void elementwiseMultiplication(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplication<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplication");
}

// elementwise division

template<typename T>
inline void elementwiseDivision(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseDivision<T>(a, b, nRows, nColumns)), "elementwiseDivision");
}

// rprop weight update

template<typename T>
inline void rpropUpdate(T *currentValues, T *newGradients, T *oldGradients, T *updateValues, T increasingFactor, T decreasingFactor,
		T maxUpdateValue, T minUpdateValue, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_rpropUpdate<T>(currentValues, newGradients, oldGradients, updateValues, increasingFactor, decreasingFactor,
			maxUpdateValue, minUpdateValue, nRows, nColumns)), "rpropUpdate");
}

// add constant elementwise

template<typename T>
inline void addConstantElementwise(T a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_addConstantElementwise<T>(a, b, nRows, nColumns)), "addConstantElementwise");
}

// elementwiseMultiplicationWithSigmoidDerivative

template<typename T>
inline void elementwiseMultiplicationWithSigmoidDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplicationWithSigmoidDerivative<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplicationWithSigmoidDerivative");
}

// elementwiseMultiplicationWithTriangleDerivative

template<typename T>
inline void elementwiseMultiplicationWithTriangleDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplicationWithTriangleDerivative<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplicationWithTriangleDerivative");
}

// elementwiseMultiplicationWithTanhDerivative

template<typename T>
inline void elementwiseMultiplicationWithTanhDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplicationWithTanhDerivative<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplicationWithTanhDerivative");
}

// multiplicationWithSoftmaxDerivative

template<typename T>
inline void multiplicationWithSoftmaxDerivative(T *a, T *b, T *c, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_multiplicationWithSoftmaxDerivative<T>(a, b, c, nRows, nColumns)),
			"multiplicationWithSoftmaxDerivative");
}

// elementwiseMultiplicationWithClipedDerivative

template <typename T>
inline void elementwiseMultiplicationWithClippedDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns, T thresholdLeft, T thresholdRight){
	CUDACALL((_cuda_elementwiseMultiplicationWithClippedDerivative<T>(a, b, nRows, nColumns, thresholdLeft, thresholdRight)),
			"elementwiseMultiplicationWithClippedDerivative");
}

// elementwiseMultiplicationWithLogDerivative

template <typename T>
inline void elementwiseMultiplicationWithLogDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplicationWithLogDerivative<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplicationWithLogDerivative");
}

// elementwiseMultiplicationWithSignedPowDerivative

template <typename T>
inline void elementwiseMultiplicationWithSignedPowDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns, T p){
	CUDACALL((_cuda_elementwiseMultiplicationWithSignedPowDerivative<T>(a, b, nRows, nColumns, p)),
			"elementwiseMultiplicationWithSignedPowDerivative");
}

// multiplicationWithL2NormalizationDerivative

template<typename T>
inline void multiplicationWithL2NormalizationDerivative(T *a, T *b, T *c, T *d, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_multiplicationWithL2NormalizationDerivative<T>(a, b, c, d, nRows, nColumns)),
			"multiplicationWithL2NormalizationDerivative");
}

// getMaxOfColumns

template<typename T>
inline void getMaxOfColumns(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_getMaxOfColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
			"getMaxOfColumns");
}


// faster getMaxOfColumns

template<typename T>
inline void getMaxOfColumns(T* vectorDevPtr, const T *matrixDevPtr, unsigned int nRows, unsigned int nColumns,
		T *tmpDevPtr, unsigned int tmpRows){
	CUDACALL((_cuda_getMaxOfColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, tmpDevPtr, tmpRows)),
			"getMaxOfColumns");
}

// addToAllColumns

template<typename T>
inline void addToAllColumns(T *a, T *b, unsigned int nRows, unsigned int nColumns, T alpha){
	CUDACALL((_cuda_addToAllColumns<T>(a, b, nRows, nColumns, alpha)),
			"addToAllColumns");
}
// addToAllChannels
// Adds one element of vector to one channel
template<typename T>
inline void addToAllChannels(T *a, T *b, unsigned int channels, unsigned int nRows, unsigned int nColumns, T alpha) {
	CUDACALL((_cuda_addToAllChannels<T>(a, b, channels, nRows, nColumns, alpha)),
			"addToAllChannels");
}
// addToAllRows

template<typename T>
inline void addToAllRows(T *a, T *b, unsigned int nRows, unsigned int nColumns, T alpha){
	CUDACALL((_cuda_addToAllRows<T>(a, b, nRows, nColumns, alpha)),
			"addToAllRows");
}

// multiplyColumnsByScalars

template<typename T>
inline void multiplyColumnsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_multiplyColumnsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
			"multiplyColumnsByScalars");
}

// divideColumnsByScalars

template<typename T>
inline void divideColumnsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_divideColumnsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
			"divideColumnsByScalars");
}

// multiplyRowsByScalars

template<typename T>
inline void multiplyRowsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_multiplyRowsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
			"multiplyRowsByScalars");
}

// divideRowsByScalars

template<typename T>
inline void divideRowsByScalars(const T *vectorDevPtr, T *matrixDevPtr, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_divideRowsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
			"divideRowsByScalars");
}
// convolution
template<typename T>
inline void prepareConvolution(T* dest, const T* source, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 destRows, const u32 destCols,
		const u32 strideX, const u32 strideY) {
	CUDACALL((_cuda_prepareConvolution<T>(dest, source, sourceWidth, sourceHeight, sourceChannels,
			kernelWidth, kernelHeight, destRows, destCols, strideX, strideY)), "prepareConvolution");
}
template<typename T>
inline void prepareConvolutionBackProp(T* dest, const T* source, const u32 destWidth, const u32 destHeight,
		const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 destRows, const u32 destCols) {
	CUDACALL((_cuda_prepareConvolutionBackProp<T>(dest, source, destWidth, destHeight,
			destChannels, kernelWidth, kernelHeight, destRows, destCols)), "prepareConvolutionBackProp");
}
template<typename T>
inline void prepareConvolutionSame(T* dest, const T* source, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 destRows,
		const u32 destCols, const u32 strideX, const u32 strideY) {
	CUDACALL((_cuda_prepareConvolutionSame<T>(dest, source, sourceWidth, sourceHeight, sourceChannels,
			kernelWidth, kernelHeight, destRows, destCols, strideX, strideY)), "prepareConvolutionSame");
}
template<typename T>
inline void prepareConvolutionSameBackProp(T* dest, const T* source, const u32 destWidth, const u32 destHeight,
		const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 destRows,
		const u32 destCols, const u32 strideX, const u32 strideY) {
	CUDACALL((_cuda_prepareConvolutionSameBackProp<T>(dest, source, destWidth, destHeight,
				destChannels, kernelWidth, kernelHeight, destRows, destCols, strideX, strideY)), "prepareConvolutionSameBackProp");
}
template<typename T>
inline void rearrange(T *dest, const T *source, const u32 sourceRows, const u32 destRows, const u32 destColumns,
		const u32 destNumPixels) {
	CUDACALL((_cuda_rearrange<T>(dest, source, sourceRows, destRows, destColumns, destNumPixels)), "reArange");
}
template<typename T>
inline void rearrangeBackProp(T *dest, const T *source, const u32 sourceColumns, const u32 destRows,
		const u32 destColumns, const u32 numPixels) {
	CUDACALL((_cuda_rearrangeBackProp(dest, source, sourceColumns, destRows, destColumns, numPixels)), "reArangeBackProp");
}
// avgPooling
template<typename T>
inline void avgPool(const T *matrix, T *result, const u32 sourceRows, const u32 sourceColumns, const u32 sourceWidth,
		const u32 sourceHeight, const u32 sourceChannels, const u32 poolSize, const u32 stride) {
	CUDACALL((_cuda_avgPool<T>(matrix, result,
			sourceRows, sourceColumns, sourceWidth, sourceHeight, sourceChannels, poolSize, stride)), "avgPool");
}
template<typename T>
inline void backPropogateAvgPool(T *result, const T *errorSignalOut , const u32 sourceRows, const u32 sourceColumns,
		const u32 sourceWidth, const u32 sourceHeight, const u32 sourceChannels, const u32 poolSize, const u32 stride) {
	CUDACALL((_cuda_backPropogateAvgPool(result, errorSignalOut, sourceRows, sourceColumns,
			sourceWidth, sourceHeight, sourceChannels, poolSize, stride)),"backPropogateAvgPool");
}
// maxPooling
template<typename T>
inline void maxPool(const T *matrix, T *result, const u32 sourceRows, const u32 sourceColumns, const u32 sourceWidth,
		const u32 sourceHeight, const u32 sourceChannels, const u32 poolSize, const u32 stride) {
	CUDACALL((_cuda_maxPool<T>(matrix, result,
			sourceRows, sourceColumns, sourceWidth, sourceHeight, sourceChannels, poolSize, stride)), "maxPool");
}
template<typename T>
inline void backPropogateMaxPool(T *result, const T *activationIn, const T *activationOut, const T *errorSignal,
		const u32 sourceRows, const u32 sourceColumns,const u32 sourceWidth, const u32 sourceHeight, const u32 sourceChannels,
		const u32 poolSize, const u32 stride) {
	CUDACALL((_cuda_backPropogateMaxPool<T>(result, activationIn, activationOut, errorSignal, sourceRows, sourceColumns,
			sourceWidth, sourceHeight, sourceChannels, poolSize, stride)),"backPropogateMaxPool");
}

// fill

template<typename T>
inline void fill(T *devPtr, T value, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_fill<T>(devPtr, value, nRows, nColumns)), "fill");
}

// ensure minimal value

template<typename T>
inline void ensureMinimalValue(T *devPtr, T value, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_ensureMinimalValue<T>(devPtr, value, nRows, nColumns)),
			"ensureMinimalValue");
}

// ensure maximal value

template<typename T>
inline void ensureMaximalValue(T *devPtr, T value, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_ensureMaximalValue<T>(devPtr, value, nRows, nColumns)),
			"ensureMaximalValue");
}

// return index of maximal value

template<typename T>
inline void argMax(T *devPtr, unsigned int nRows, unsigned int nColumns, unsigned int *resultDev) {
	CUDACALL((_cuda_argMax<T>(devPtr, nRows, nColumns, resultDev)),
			"argMax");
}

// set maximal element in each column to 1.0, all other elements to zero

template<typename T>
inline void max(T *devResult, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_max<T>(devResult, nRows, nColumns)),
			"max");
}

// elementwise maximum: devResult = max(devA, devB)

template<typename T>
inline void max(T *devResult, const T *devA, const T *devB, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_max<T>(devResult, devA, devB, nRows, nColumns)),
			"max");
}

// elementwise Kronecker delta: devResult = \delta(devA, devB)

template<typename T>
inline void elementwiseMultiplicationWithKroneckerDelta(T *devResult, const T *devA, const T *devB, unsigned int nRows, unsigned int nColumns) {
	CUDACALL((_cuda_elementwiseMultiplicationWithKroneckerDelta<T>(devResult, devA, devB, nRows, nColumns)),
			"elementwiseMultiplicationWithKroneckerDelta");
}

// number of classification errors

template<typename T>
inline void nClassificationErrors(T *devPtr, unsigned int nRows, unsigned int nColumns, T *targets, unsigned int *resultDev) {
	CUDACALL((_cuda_nClassificationErrors<T>(devPtr, nRows, nColumns, targets, resultDev)),
			"nClassificationErrors");
}

// cross-entropy objective function
template<typename T>
inline void crossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev){
	CUDACALL((_cuda_crossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, resultDev)),
			"crossEntropyObjectiveFunction");
}

// weighted cross-entropy objective function
template<typename T>
inline void weightedCrossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev, T *weights){
	CUDACALL((_cuda_weightedCrossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, resultDev, weights)),
			"weightedCrossEntropyObjectiveFunction");
}

// squared error objective function
template<typename T>
inline void squaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev){
	CUDACALL((_cuda_squaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, resultDev)),
			"squaredErrorObjectiveFunction");
}

// weighted squared error objective function
template<typename T>
inline void weightedSquaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev, T *weights){
	CUDACALL((_cuda_weightedSquaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, resultDev, weights)),
			"weightedSquaredErrorObjectiveFunction");
}

// smoothed L1 objective function
template<typename T>
inline void smoothedL1ObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T *resultDev){
	CUDACALL((_cuda_smoothedL1ObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, resultDev)),
			"smoothedL1ObjectiveFunction");
}

// weighted smoothed L1 objective function
template<typename T>
inline void weightedSmoothedL1ObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, T *targets, T* weights, T *resultDev){
	CUDACALL((_cuda_weightedSmoothedL1ObjectiveFunction<T>(matrixPtr, nRows, nColumns, targets, weights, resultDev)),
			"weightedSmoothedL1ObjectiveFunction");
}

// second order features
template<typename T>
inline void appendSecondOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset){
	CUDACALL((_cuda_appendSecondOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
			"appendSecondOrderFeatures");
}

// diagonal second order features
template<typename T>
inline void appendDiagonalSecondOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset){
	CUDACALL((_cuda_appendDiagonalSecondOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
			"appendDigonalSecondOrderFeatures");
}

// third order features
template<typename T>
inline void appendThirdOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset){
	CUDACALL((_cuda_appendThirdOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
			"appendThirdOrderFeatures");
}

// third order features
template<typename T>
inline void appendDiagonalThirdOrderFeatures(const T *X, unsigned int nRowsX, unsigned int nColumnsX, T *Y, unsigned int nRowsY, unsigned int offset){
	CUDACALL((_cuda_appendDiagonalThirdOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
			"appendDiagonalThirdOrderFeatures");
}

// gaussian mixture posteriors (not normalized, not exponentiated -> p(c|x) is obtain if softmax is applied to result)
template<typename T>
inline void gaussianMixturePosteriors(T *P, const T *X, const T *means, const T *variances, const T *weights,
		unsigned int nFeatures, unsigned int featureDim, unsigned int nMixtures){
	CUDACALL((_cuda_gaussianMixturePosteriors<T>(P, X, means, variances, weights, nFeatures, featureDim, nMixtures)), "gaussianMixturePosteriors");
}

// fisher encoding of each column
template<typename T>
inline void fisherEncoding(T *F, const T *X, const T *means, const T *variances, const T *weights, const T *gamma,
		unsigned int nFeatures, unsigned int featureDim, unsigned int nMixtures){
	CUDACALL((_cuda_fisherEncoding<T>(F, X, means, variances, weights, gamma, nFeatures, featureDim, nMixtures)), "fisherEncoding");
}

// dropout
template<typename T>
inline void dropout(T *X, const T *mask, unsigned int nRows, unsigned int nColumns, T dropoutProbability){
	CUDACALL((_cuda_dropout<T>(X, mask, nRows, nColumns, dropoutProbability)), "dropout");
}

} // namespace Cuda

} // namespace Math

#endif /* MATH_CUDAMATRIXKERNELSWRAPPER_HH_ */
