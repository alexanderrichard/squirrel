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
// Used with permission by RWTH Aachen University.
#include "CudaDataStructure.hh"
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include "Core/Configuration.hh"

using namespace Math;

const Core::ParameterBool CudaDataStructure::paramUseGpu_("use-gpu", true, "");

bool CudaDataStructure::initialized = false;

u32 CudaDataStructure::deviceUsedMemory = 0;

u32 CudaDataStructure::peakMemoryValue = 0;

bool CudaDataStructure::_hasGpu = false;

cublasHandle_t CudaDataStructure::cublasHandle;

curandGenerator_t CudaDataStructure::randomNumberGenerator;

void CudaDataStructure::initialize(){
	if (initialized)
		return;
	initialized = true;
	Core::Log::openTag("cuda");
	// check whether MODULE_CUDA is active and a GPU is available
	int nGpus = 0;
	bool hasCuda = false;
	int success;
	if (Core::Configuration::config(paramUseGpu_)) {
		success = Cuda::getNumberOfGpus(nGpus, hasCuda);
	}
	if (hasCuda){
		if (success != 0){ // no GPU available, or some error occured
			std::ostringstream ss; ss << "Using binary with GPU support, but no GPU available. Error code is: " << success;
			warning(ss.str());
		}

		_hasGpu = nGpus > 0;
		// initialize cuBLAS and cuRAND
		if (_hasGpu){
			std::ostringstream ss; ss << "using 1 of " << nGpus << " GPUs";
			log(ss.str());
			success = Cuda::createCublasHandle(cublasHandle);
			if (success != 0){
				std::string msg = "Failed to initialize cuBLAS library";
				criticalError(msg);
			}
			success = Cuda::createRandomNumberGenerator(randomNumberGenerator, CURAND_RNG_PSEUDO_DEFAULT);
			if (success != 0){
				std::string msg = "Failed to initialize cuRAND random number generator library";
				criticalError(msg);
			}
			success = Cuda::setSeed(randomNumberGenerator, (unsigned long long) rand());
			if (success != 0){
				std::string msg = "Failed to set seed for cuRAND random number generator";
				criticalError(msg);
			}
		}
		// this should never occur (when no GPU is available, a non-zero error code is returned)
		if (!_hasGpu && success == 0) {
			std::string msg = "Using binary with GPU support, but no GPU available.";
			warning(msg);
		}
	}
	else { // if !hasCuda
		log("CUDA disabled. Use CPU.");
	}
	Core::Log::closeTag();
}

bool CudaDataStructure::hasGpu(){
	if (!initialized)
		initialize();
	return _hasGpu;
}
CudaDataStructure::CudaDataStructure() :
		gpuMode_(hasGpu())
{ }

CudaDataStructure::CudaDataStructure(const CudaDataStructure &x ) :
		gpuMode_(x.gpuMode_)
{ }

void CudaDataStructure::log(const std::string &msg){
	Core::Log::os() << msg;
}

void CudaDataStructure::warning(const std::string &msg){
	std::cerr << msg << std::endl;
}


void CudaDataStructure::error(const std::string &msg){
	std::cerr << msg << std::endl;
	exit(1);
}

void CudaDataStructure::criticalError(const std::string &msg){
	std::cerr << msg << std::endl;
	exit(1);
}
