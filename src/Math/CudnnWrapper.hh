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

/*
 * CudnnWrapper.hh
 *
 *  Created on: Aug 10, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNWRAPPER_HH_
#define MATH_CUDNNWRAPPER_HH_

#include <iostream>
#include <stdlib.h>
#include <Modules.hh>

#ifdef MODULE_CUDNN
#include <cudnn.h>
#endif


namespace Math {
#ifndef MODULE_CUDNN

static int CUDNN_STATUS_SUCCESS = 0;
typedef int cudnnStatus_t;
typedef int cudnnDataType_t;

struct cudnnHandle_t {
	int dummyHandle;
};
struct cudnnTensorFormat_t {};
struct cudnnTensorDescriptor_t {};
struct cudnnFilterDescriptor_t {};
struct cudnnConvolutionDescriptor_t {};
struct cudnnConvolutionMode_t {};
struct cudnnPoolingDescriptor_t {};
#endif

namespace cuDNN {
inline cudnnStatus_t cuDNNCreateHandle(cudnnHandle_t &cudnnHandle){
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnCreate(&cudnnHandle);
#else
	std::cerr<<"Calling cudnn method 'cuDNNCreateHandle' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNCreateTensorDescriptor(cudnnTensorDescriptor_t &cudnnTensorDesc) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnCreateTensorDescriptor(&cudnnTensorDesc);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Failed to create cuDNN Source Tensor Descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNCreateTensorDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNCreateFilterDescriptor(cudnnFilterDescriptor_t &cudnnFilterDesc) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnCreateFilterDescriptor(&cudnnFilterDesc);
	if ( result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Failed to create cuDNN Filter Descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNCreateFilterDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNSetFilterDescriptor(cudnnFilterDescriptor_t &filterDesc, cudnnDataType_t dataType,
		cudnnTensorFormat_t tensorFormat, int outputFeatures, int inputFeatures, int filterHeight, int filterWidth) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	int filterDim[4] = {outputFeatures, inputFeatures, filterHeight, filterWidth};
	//result = cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat, outputFeatures, inputFeatures, filterHeight, filterWidth);
	result = cudnnSetFilterNdDescriptor(filterDesc, dataType, tensorFormat, 4, filterDim);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to set Filter Description! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNSetFilterDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t &convDesc) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnCreateConvolutionDescriptor(&convDesc);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Failed to create cuDNN Convolution Descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNCreateConvolutionDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNSetConvolutionDescriptor(cudnnConvolutionDescriptor_t &convDesc, int padHeight, int padWidth,
		int verticalStride, int horizontalStride, int upscaleX, int upscaleY, cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	//result = cudnnSetConvolution2dDescriptor(convDesc, padHeight, padWidth, verticalStride, horizontalStride, upscaleX, upscaleY, mode);
	int pad[2] = {padHeight, padWidth};
	int filterStride[2] = {verticalStride, horizontalStride};
	int upScale[2] = {upscaleY, upscaleX};
	cudnnSetConvolutionNdDescriptor(convDesc, 2, pad, filterStride, upScale, mode, dataType);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Failed to set Convolution Descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNSetConvolutionDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNGetConvolutionNdForwardOutputDim(cudnnConvolutionDescriptor_t &convDesc, cudnnTensorDescriptor_t &srcTensorDesc,
		cudnnFilterDescriptor_t &filterDesc, int &outBatch, int	&outChannels, int &outHeight, int &outWidth) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	int tensorOuputDim[4] = {0,0,0,0};
	result = cudnnGetConvolutionNdForwardOutputDim(convDesc,srcTensorDesc, filterDesc, 4, tensorOuputDim);
	outBatch = tensorOuputDim[0];
	outChannels = tensorOuputDim[1];
	outHeight = tensorOuputDim[2];
	outWidth = tensorOuputDim[3];
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Failed to get Convolution Forward output dimension! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNGetConvolution2dForwardOutputDim' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNCreatePoolingDescriptor(cudnnPoolingDescriptor_t &poolingDesc) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnCreatePoolingDescriptor(&poolingDesc);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to create pooling descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNCreatePoolingDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNSetPoolingDescriptor(cudnnPoolingDescriptor_t &poolingDesc,
		cudnnPoolingMode_t mode, int gridSize, int padX, int padY, int strideSize) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	int grid[2] = {gridSize, gridSize};
	int pad[2] = {padY, padX};
	int stride[2] = {strideSize, strideSize};
	result = cudnnSetPoolingNdDescriptor(poolingDesc, mode, CUDNN_PROPAGATE_NAN, 2, grid, pad, stride);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to set pooling descriptor! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNSetPoolingDescriptor' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNGetPoolingNdForwardOutputDim(cudnnPoolingDescriptor_t &poolingDesc,
		cudnnTensorDescriptor_t &srcTensorDesc, int outputDim[]) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnGetPoolingNdForwardOutputDim(poolingDesc, srcTensorDesc, 4, outputDim);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to get pooling output dimension! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNGetPoolingNdForwardOutputDim' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNSetTensorDescription(cudnnTensorDescriptor_t &tesnorDesc,
		cudnnDataType_t &dataType, int numBatches, int channels, int height, int width) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	int dim[4] = {numBatches, channels, height, width};
	int strides[4] = {channels * height * width, height * width, width, 1};
	result = cudnnSetTensorNdDescriptor(tesnorDesc, dataType, 4, dim, strides);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to set Tensor Description! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNSetTensorDescription' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNGetConvolutionForwardAlgorithm(cudnnHandle_t &handle, const cudnnTensorDescriptor_t &srcDesc,
		const cudnnFilterDescriptor_t &filterDesc, const cudnnConvolutionDescriptor_t &convDesc, const cudnnTensorDescriptor_t &destDesc,
		cudnnConvolutionFwdPreference_t preference, size_t memoryLimitInbytes, cudnnConvolutionFwdAlgo_t &algo) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnGetConvolutionForwardAlgorithm(handle, srcDesc, filterDesc, convDesc, destDesc,
			preference, memoryLimitInbytes, &algo);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to Get Forward Algorithm! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cudnnGetConvolutionForwardAlgorithm' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNGetConvolutionBackwardDataAlgorithm(cudnnHandle_t &handle, const cudnnFilterDescriptor_t &filterDesc,
		const cudnnTensorDescriptor_t &destDesc, const cudnnConvolutionDescriptor_t &convDesc, const cudnnTensorDescriptor_t &srcDesc,
		cudnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInbytes, cudnnConvolutionBwdDataAlgo_t &algo) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnGetConvolutionBackwardDataAlgorithm(handle, filterDesc, destDesc, convDesc, srcDesc,
			preference, memoryLimitInbytes, &algo);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to Get Backward Data Algorithm! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNGetConvolutionBackwardDataAlgorithm' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
inline cudnnStatus_t cuDNNGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t &handle, const cudnnTensorDescriptor_t &srcDesc,
		const cudnnTensorDescriptor_t &destDesc, const cudnnConvolutionDescriptor_t &convDesc, const cudnnFilterDescriptor_t &filterDesc,
		cudnnConvolutionBwdFilterPreference_t preference, size_t memoryLimitInbytes, cudnnConvolutionBwdFilterAlgo_t &algo) {
	cudnnStatus_t result = CUDNN_STATUS_SUCCESS;
#ifdef MODULE_CUDNN
	result = cudnnGetConvolutionBackwardFilterAlgorithm(handle, srcDesc, destDesc, convDesc,
			filterDesc, preference, memoryLimitInbytes, &algo);
	if (result != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to Get Backward Filter Algorithm! cuDNNStatus:"<<result<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNGetConvolutionBackwardFilterAlgorithm' in binary without cudnn support!"<<std::endl;
#endif
	return result;
}
template<typename T>
inline cudnnDataType_t cuDNNInferDataType() {
	cudnnDataType_t result;
#ifdef MODULE_CUDNN
	switch(sizeof(T)) {
	case 4:
		result = CUDNN_DATA_FLOAT;
		break;
	case 8:
		result = CUDNN_DATA_DOUBLE;
		break;
	default:
		std::cerr<<"Unsupported cudnn data type!"<<std::endl;
		exit(1);
	}
#else
	std::cerr<<"Calling cudnn method 'cuDNNInferDataType' in binary without cudnn support!"<<std::endl;
	exit(1);
#endif
	return result;
}

}//namespace cuDNN

} //namespace Math




#endif /* MATH_CUDNNWRAPPER_HH_ */
