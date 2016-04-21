// Copyright 2011 RWTH Aachen University. All rights reserved.
//
// Licensed under the RWTH ASR License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Used with permission by RWTH University.
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

// elementwiseMultiplicationWithRectifiedDerivative

template <typename T>
inline void elementwiseMultiplicationWithRectifiedDerivative(T *a, T *b, unsigned int nRows, unsigned int nColumns){
	CUDACALL((_cuda_elementwiseMultiplicationWithRectifiedDerivative<T>(a, b, nRows, nColumns)),
			"elementwiseMultiplicationWithRectifiedDerivative");
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

// return index of maximal value

template<typename T>
inline void argMax(T *devPtr, unsigned int nRows, unsigned int nColumns, unsigned int *resultDev) {
	CUDACALL((_cuda_argMax<T>(devPtr, nRows, nColumns, resultDev)),
			"argMax");
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
inline void nClassificationErrors(T *devPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, unsigned int *resultDev) {
	CUDACALL((_cuda_nClassificationErrors<T>(devPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
			"nClassificationErrors");
}

// cross-entropy objective function
template<typename T>
inline void crossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev){
	CUDACALL((_cuda_crossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
			"crossEntropyObjectiveFunction");
}

// weighted cross-entropy objective function
template<typename T>
inline void weightedCrossEntropyObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev, T *weights){
	CUDACALL((_cuda_weightedCrossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
			"weightedCrossEntropyObjectiveFunction");
}

// squared error objective function
template<typename T>
inline void squaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev){
	CUDACALL((_cuda_squaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
			"squaredErrorObjectiveFunction");
}
// weighted squared error objective function
template<typename T>
inline void weightedSquaredErrorObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev, T *weights){
	CUDACALL((_cuda_weightedSquaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
			"weightedSquaredErrorObjectiveFunction");
}

// binary divergence objective function
template <typename T>
inline void binaryDivergenceObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev) {
	CUDACALL((_cuda_binaryDivergenceObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
			"binaryDivergenceObjectiveFunction");
}

template <typename T>
inline void weightedBinaryDivergenceObjectiveFunction(T *matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int *alignmentDevPtr, T *resultDev, T *weights){
	CUDACALL((_cuda_weightedBinaryDivergenceObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
			"weightedBinaryDivergenceObjectiveFunction");
}

template <typename T>
inline void binaryDivergenceSoftmaxGradient(T *matrixPtr, unsigned int nRows, unsigned int nColumns, const T *outputDevPtr, const unsigned int *alignmentDevPtr){
	CUDACALL((_cuda_binaryDivergenceSoftmaxGradient<T>(matrixPtr, nRows, nColumns, outputDevPtr, alignmentDevPtr)),
			"binaryDivergenceSoftmaxGradient");
}

// Kronecker Delta
template<typename T>
inline void addKroneckerDelta(T *matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int *alignmentDevPtr, const T scale){
	CUDACALL((_cuda_addKroneckerDelta<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, scale)),
			"addKroneckerDelta");
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
