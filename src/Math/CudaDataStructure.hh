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
#ifndef MATH_CUDADATASTRUCTURE_HH_
#define MATH_CUDADATASTRUCTURE_HH_

#include <Math/CudaWrapper.hh>
#include <Math/CublasWrapper.hh>
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
