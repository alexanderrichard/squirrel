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

 /* CudnnDataStructure.cc
 *
 *  Created on: Sep 1, 2016
 *      Author: ahsan
 */
#include <Core/CommonHeaders.hh>
#include "CudnnDataStructure.hh"
#include "CudaDataStructure.hh"

using namespace Math;
using namespace Math::cuDNN;

bool CudnnDataStructure::isInitialized_ = false;
cudnnHandle_t CudnnDataStructure::cudnnHandle_;
#ifdef MODULE_CUDNN
cudnnTensorFormat_t CudnnDataStructure::tensorFormat_ = CUDNN_TENSOR_NCHW;
#endif

void CudnnDataStructure::initialize() {
	if(CudaDataStructure::hasGpu() && !isInitialized_) {
		cudnnStatus_t cudnnStatus;
		cudnnStatus = cuDNN::cuDNNCreateHandle(cudnnHandle_);
		if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
			std::cerr<<"Failed to create cuDNN handle"<<std::endl;
			exit(1);
		}
		isInitialized_ = true;
	}
}


/*
*/
