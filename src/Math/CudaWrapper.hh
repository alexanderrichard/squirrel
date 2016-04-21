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
#ifndef MATH_CUDAWRAPPER_HH_
#define MATH_CUDAWRAPPER_HH_


#include <Modules.hh>
#include <stdlib.h>
#include <iostream>

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#endif

/*
 * wrapper for CUDA routines
 */


namespace Math {

#ifndef MODULE_CUDA
struct curandGenerator_t {
	int dummyGenerator;
};
typedef int curandRngType_t;
#define CURAND_RNG_PSEUDO_DEFAULT 0
#endif

namespace Cuda {

#ifndef MODULE_CUDA
typedef int cudaError_t;
#endif

inline int getNumberOfGpus(int &count, bool &hasCuda){
	int success = 0;
	hasCuda = false;
	count = 0;
#ifdef MODULE_CUDA
	success = cudaGetDeviceCount(&count);
	hasCuda = true;
#endif
	return success;
}

inline int createRandomNumberGenerator(curandGenerator_t &generator, curandRngType_t rng_type){
	int result = 0;
#ifdef MODULE_CUDA
	result = curandCreateGenerator(&generator, rng_type);
#else
	std::cerr << "Calling gpu method 'createRandomNumberGenerator' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

inline int setSeed(curandGenerator_t &generator, unsigned long long seed){
	int result = 0;
#ifdef MODULE_CUDA
	result = curandSetPseudoRandomGeneratorSeed(generator, seed);
#else
	std::cerr << "Calling gpu method 'setSeed' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

inline unsigned int cudaDeviceSync(bool hasGpu=true){
	int result = 0;
#ifdef MODULE_CUDA
	result = hasGpu ? cudaDeviceSynchronize() : true;
#endif
	return result;
}

inline void printError(cudaError_t err){
#ifdef MODULE_CUDA
	std::cout << "Error:\t" << cudaGetErrorString(err) << std::endl;
#else
	std::cout << "Error:\t" << err << std::endl;
#endif
}

inline int getMemoryInfo(unsigned int* free, unsigned int* total) {
	int result = 0;
#ifdef MODULE_CUDA
	cudaError_t err = cudaMemGetInfo((size_t*)free, (size_t*)total);
	result = err;
#else
	std::cerr << "Calling gpu method 'gpuGetMemoryInfo' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
int alloc(T *&devPtr, size_t nElements){
	int result = 0;
#ifdef MODULE_CUDA
	cudaError_t err = cudaMalloc((void **) &devPtr, nElements * sizeof(T));
	result = err;
#else
	std::cerr << "Calling gpu method 'gpuAlloc' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
int free(T *devPtr){
	int result = 0;;
#ifdef MODULE_CUDA
	result = cudaFree((void *) devPtr);
#else
	std::cerr << "Calling gpu method 'gpuFree' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
int copyFromGpu(T *dst, const T* src, size_t nElements){
	int result = 0;;
#ifdef MODULE_CUDA
	result = cudaMemcpy(dst, src, nElements * sizeof(T), cudaMemcpyDeviceToHost);
#else
	std::cerr << "Calling gpu method 'cppyFromGpu' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
int copyToGpu(T *dst, const T* src, size_t nElements){
	int result = 0;;
#ifdef MODULE_CUDA
	result = cudaMemcpy(dst, src, nElements * sizeof(T), cudaMemcpyHostToDevice);
#else
	std::cerr << "Calling gpu method 'copyToGpu' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
int memSet(T * devPtr, int value, size_t count){
	int result = 0;;
#ifdef MODULE_CUDA
	result = cudaMemset(devPtr, value, count * sizeof(T));
#else
	std::cerr << "Calling gpu method 'memSet' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
inline int generateUniform(curandGenerator_t &generator, T *outputPtr, size_t num);

template<>
inline int generateUniform(curandGenerator_t &generator, float *outputPtr, size_t num) {
	int result = 0;
#ifdef MODULE_CUDA
	result = curandGenerateUniform(generator, outputPtr, num);
#else
	std::cerr << "Calling gpu method 'generateUniform' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<>
inline int generateUniform(curandGenerator_t &generator, double *outputPtr, size_t num) {
	int result = 0;
#ifdef MODULE_CUDA
	result = curandGenerateUniformDouble(generator, outputPtr, num);
#else
	std::cerr << "Calling gpu method 'generateUniform' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<typename T>
inline int generateNormal(curandGenerator_t &generator, T *outputPtr, size_t num, T mean, T stddev);

template<>
inline int generateNormal(curandGenerator_t &generator, float *outputPtr, size_t num, float mean, float stddev) {
	int result = 0;
#ifdef MODULE_CUDA
	result = curandGenerateNormal(generator, outputPtr, num, mean, stddev);
#else
	std::cerr << "Calling gpu method 'generateNormal' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

template<>
inline int generateNormal(curandGenerator_t &generator, double *outputPtr, size_t num, double mean, double stddev) {
	int result = 0;
#ifdef MODULE_CUDA
	result = curandGenerateNormalDouble(generator, outputPtr, num, mean, stddev);
#else
	std::cerr << "Calling gpu method 'generateNormal' in binary without gpu support!" << std::endl;
	exit(1);
#endif
	return result;
}

} // namespace Cuda

} // namespace Math


#endif /* MATH_CUDAWRAPPER_HH_ */
