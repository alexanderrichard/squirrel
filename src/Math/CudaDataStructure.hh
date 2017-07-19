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

#ifndef MATH_CUDADATASTRUCTURE_HH_
#define MATH_CUDADATASTRUCTURE_HH_

#include <Math/CudaWrapper.hh>
#include <Math/CublasWrapper.hh>
#include <Math/CudnnWrapper.hh>
#include "Core/Log.hh"

namespace Math {

/*
 * CudaDataStructure
 *
 * base class for all data structures that require initialization of Cuda resources
 * availability of GPUs is checked here once
 * cublas handle is created here once
 *
 */

class CudaDataStructure  {
private:
	static const Core::ParameterBool paramUseGpu_;
	static bool initialized;
protected:
	static u32 deviceUsedMemory;
	static u32 peakMemoryValue;
private:
	static bool _hasGpu;
protected:
	const bool gpuMode_;
protected:

	static cublasHandle_t cublasHandle;
	// create a single random number generator
	static curandGenerator_t randomNumberGenerator;
	static void initialize();
public:
	static bool hasGpu();
public:
	// constructor with memory allocation
	CudaDataStructure();
	CudaDataStructure(const CudaDataStructure &x);
protected:
	static void log(const std::string &msg);
	static void warning(const std::string &msg);
	static void error(const std::string &msg);
	static void criticalError(const std::string &msg);
};

} // namespace Math

#endif /* MATH_CUDADATASTRUCTURE_HH_ */
