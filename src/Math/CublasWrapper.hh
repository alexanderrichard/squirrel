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

#ifndef MATH_CUBLASWRAPPER_HH_
#define MATH_CUBLASWRAPPER_HH_

#include <stdlib.h>
#include <iostream>
#include <Modules.hh>
#include <Core/Utils.hh>
#include <Math/CudaWrapper.hh>

#ifdef MODULE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "CudaMatrixKernels.hh"
#endif

/*
 * wrapper for CUDA and CuBLAS routines
 */

namespace Math {

#ifndef MODULE_CUDA
struct cublasHandle_t {
	int dummyHandle;
};
#endif

namespace Cuda {

inline int createCublasHandle(cublasHandle_t &handle){
	int result = 0;;
#ifdef MODULE_CUDA
	result = cublasCreate(&handle);
#else
	std::cerr << "Calling gpu method 'createCublasHandle' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T, typename S>
inline int axpy(cublasHandle_t handle, int n, S alpha, S *x, int incx, T *y, int incy);

template<>
inline int axpy(cublasHandle_t handle, int n, float alpha, float *x, int incx, float *y, int incy){
	int result = 0;
#ifdef MODULE_CUDA
	result = cublasSaxpy(handle, n, &alpha, x, incx, y, incy);
#else
	std::cerr << "Calling gpu method 'axpy' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<>
inline int axpy(cublasHandle_t handle, int n, double alpha, double *x, int incx, double *y, int incy){
	int result = 0;
#ifdef MODULE_CUDA
	result = cublasDaxpy(handle, n, &alpha, x, incx, y, incy);
#else
	std::cerr << "Calling gpu method 'axpy' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

// own kernel for mixed precision
template<>
inline int axpy(cublasHandle_t handle, int n, float alpha, float *x, int incx, double *y, int incy){
	require_eq(incx, 1); // general case not implemented yet
	require_eq(incy, 1);
#ifdef MODULE_CUDA
	_cuda_axpy(n, alpha, x, y);
#else
	std::cerr << "Calling gpu method 'axpy' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return 0; // no check
}

// mixed precision implementation is in CudaMatrixKernelsWrapper.hh

template<typename T>
inline int dot(cublasHandle_t handle, int n, const T *x, int incx, const T *y, int incy, T &result);

template<>
inline int dot(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float &result){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasSdot(handle, n, x, incx, y, incy, &result);
#else
	std::cerr << "Calling gpu method 'dot' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<>
inline int dot(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double &result){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasDdot(handle, n, x, incx, y, incy, &result);
#else
	std::cerr << "Calling gpu method 'dot' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<typename T>
inline int scal(cublasHandle_t handle, int n, T alpha, T *x, int incx);

template<>
inline int scal(cublasHandle_t handle, int n, float alpha, float *x, int incx){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasSscal(handle, n, &alpha, x, incx);
#else
	std::cerr << "Calling gpu method 'scale' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<>
inline int scal(cublasHandle_t handle, int n, double alpha, double *x, int incx){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasDscal(handle, n, &alpha, x, incx);
#else
	std::cerr << "Calling gpu method 'scale' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<typename T>
inline int copy(cublasHandle_t handle, int n, const T *x, int incx, T *y, int incy);

template<>
inline int copy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasScopy(handle, n, x, incx, y, incy);
#else
	std::cerr << "Calling gpu method 'copy' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<>
inline int copy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
	int cublasStatus = 0;
#ifdef MODULE_CUDA
	cublasStatus = cublasDcopy(handle, n, x, incx, y, incy);
#else
	std::cerr << "Calling gpu method 'copy' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return cublasStatus;
}

template<typename T>
inline int asum(cublasHandle_t handle, int n, const T *x, int incx, T *result);

template<>
inline int asum(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasSasum(handle, n, x, incx, result);
#else
	std::cerr << "Calling gpu method 'asum' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<>
inline int asum(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasDasum(handle, n, x, incx, result);
#else
	std::cerr << "Calling gpu method 'asum' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<typename T>
inline int nrm2(cublasHandle_t handle, int n, const T *x, int incx, T *result);

template<>
inline int nrm2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasSnrm2(handle, n, x, incx, result);
#else
	std::cerr << "Calling gpu method 'nrm2' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<>
inline int nrm2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasDnrm2(handle, n, x, incx, result);
#else
	std::cerr << "Calling gpu method 'nrm2' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<typename T>
inline int iamax(cublasHandle_t handle, int n, const T *x, int incx, int *result);

template<>
inline int iamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasIsamax(handle, n, x, incx, result);
	// for some compatibility reason, this function returns 1-based indices
	*result = *result - 1;
#else
	std::cerr << "Calling gpu method 'amax' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<>
inline int iamax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasIdamax(handle, n, x, incx, result);
	// for some compatibility reason, this function returns 1-based indices
	*result = *result - 1;
#else
	std::cerr << "Calling gpu method 'amax' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<typename T>
inline int iamin(cublasHandle_t handle, int n, const T *x, int incx, int *result);

template<>
inline int iamin(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasIsamin(handle, n, x, incx, result);
	// for some compatibility reason, this function returns 1-based indices
	*result = *result - 1;
#else
	std::cerr << "Calling gpu method 'amax' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<>
inline int iamin(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasIdamin(handle, n, x, incx, result);
	// for some compatibility reason, this function returns 1-based indices
	*result = *result - 1;
#else
	std::cerr << "Calling gpu method 'amax' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<typename T>
inline int ger(cublasHandle_t handle, int m, int n, const T alpha, const T *x, int incx,
		const T *y, int incy, T *A, int lda);

template<>
inline int ger(cublasHandle_t handle, int m, int n, const float alpha, const float *x, int incx,
		const float *y, int incy, float *A, int lda) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasSger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
	std::cerr << "Calling gpu method 'ger' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<>
inline int ger(cublasHandle_t handle, int m, int n, const double alpha, const double *x, int incx,
		const double *y, int incy, double *A, int lda) {
	int ret = 0;
#ifdef MODULE_CUDA
	ret = cublasDger(handle, m, n, &alpha, x, incx, y, incy, A, lda);
#else
	std::cerr << "Calling gpu method 'ger' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return ret;
}

template<typename T>
inline int gemv(cublasHandle_t handle, bool transposed, int m, int n,
		T alpha, const T *A, int lda, const T *x, int incx, T beta, T* y, int incy);

template<>
inline int gemv(cublasHandle_t handle, bool transposed, int m, int n,
		float alpha, const float *A, int lda, const float *x, int incx, float beta, float* y, int incy)
{
	int result = 0;
#ifdef MODULE_CUDA
	cublasOperation_t tr = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
	result = cublasSgemv(handle, tr, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
#else
	std::cerr << "Calling gpu method 'gemv' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<>
inline int gemv(cublasHandle_t handle, bool transposed, int m, int n,
		double alpha, const double *A, int lda, const double *x, int incx, double beta, double* y, int incy)
{
	int result = 0;
#ifdef MODULE_CUDA
	cublasOperation_t tr = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
	result = cublasDgemv(handle, tr, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
#else
	std::cerr << "Calling gpu method 'gemv' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
inline int gemm(cublasHandle_t handle, bool transposedA, bool transposedB, int m, int n, int k,
		T alpha, const T *A, int lda, const T *B, int ldb, T gamma, T *C, int ldc);

template<>
inline int gemm(cublasHandle_t handle, bool transposedA, bool transposedB, int m, int n, int k,
		float alpha, const float *A, int lda, const float *B, int ldb, float gamma, float *C, int ldc)
{
	int result = 0;
#ifdef MODULE_CUDA
	cublasOperation_t trA = transposedA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t trB = transposedB ? CUBLAS_OP_T : CUBLAS_OP_N;
	result = cublasSgemm(handle, trA, trB, m, n, k, &alpha, A, lda, B, ldb, &gamma, C, ldc);
#else
	std::cerr << "Calling gpu method 'gemm' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<>
inline int gemm(cublasHandle_t handle, bool transposedA, bool transposedB, int m, int n, int k,
		double alpha, const double *A, int lda, const double *B, int ldb, double gamma, double *C, int ldc)
{
	int result = 0;
#ifdef MODULE_CUDA
	cublasOperation_t trA = transposedA ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t trB = transposedB ? CUBLAS_OP_T : CUBLAS_OP_N;
	result = cublasDgemm(handle, trA, trB, m, n, k, &alpha, A, lda, B, ldb, &gamma, C, ldc);
#else
	std::cerr << "Calling gpu method 'gemm' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

} // namespace Cuda

} // namespace Math

#endif /* MATH_CUBLASWRAPPER_HH_ */
