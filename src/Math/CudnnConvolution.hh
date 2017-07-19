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
 * CudnnConvolution.hh
 *
 *  Created on: Sep 6, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNCONVOLUTION_HH_
#define MATH_CUDNNCONVOLUTION_HH_
#include "CudnnDataStructure.hh"

namespace Math
{

namespace cuDNN
{

template<typename T>
class CudnnConvolution : public CudnnDataStructure {
private:
	typedef CudnnDataStructure Precursor;
	cudnnTensorDescriptor_t srcTensorDescriptor_;
	cudnnTensorDescriptor_t destTensorDescriptor_;
	cudnnTensorDescriptor_t biasTensorDescriptor_;
	cudnnFilterDescriptor_t filterDescriptor_;
	cudnnConvolutionDescriptor_t convolutionDescriptor_;

	cudnnConvolutionFwdAlgo_t fwdAlgorithm_;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgorithm_;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgorithm_;


	int batchSize_;

	int sourceWidth_;
	int sourceHeight_;
	int sourceChannels_;

	int destWidth_;
	int destHeight_;
	int destChannels_;

	int padX_;
	int padY_;

	int strideX_;
	int strideY_;

	int kernelWidth_;
	int kernelHeight_;

	void createDescriptors();
	void setDescriptors();
public:
	CudnnConvolution();
	~CudnnConvolution() {}


	void init(int batchSize, int sourceWidth, int sourceHeight, int sourceChannels, int destWidth,
			int destHeight, int destChannels, int padX, int padY, int strideX, int strideY, int kernelWidth, int kernelHeight);
	void updateBatchSize(int batchSize);

	void convolveForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source, const CudaMatrix<T> &kernel);
	void convolveBackwardFilter(CudaMatrix<T> &dFilter, const CudaMatrix<T> &activationIn, const CudaMatrix<T> &errorSignalOut);
	void convolveBackwardData(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &errorSignalOut, const CudaMatrix<T> &filter);

	void addBias(T *data, const T *bias);
	void addBiasBackward(T *bias, const T *errorSignal);
};


template<typename T>
CudnnConvolution<T>::CudnnConvolution() : Precursor(),
batchSize_(0), sourceWidth_(0), sourceHeight_(0), sourceChannels_(0),
destWidth_(0), destHeight_(0), destChannels_(0), padX_(0), padY_(0),
strideX_(0), strideY_(0), kernelWidth_(0), kernelHeight_(0) { }

template<typename T>
void CudnnConvolution<T>::init(int batchSize, int sourceWidth, int sourceHeight, int sourceChannels, int destWidth,
		int destHeight, int destChannels, int padX, int padY, int strideX, int strideY, int kernelWidth, int kernelHeight) {

	batchSize_ = batchSize;
	sourceHeight_ = sourceHeight;
	sourceWidth_ = sourceWidth;
	sourceChannels_ = sourceChannels;
	destWidth_ = destWidth;
	destHeight_ = destHeight;
	destChannels_ = destChannels;
	padX_ = padX;
	padY_ = padY;
	strideX_ = strideX;
	strideY_ = strideY;
	kernelWidth_ = kernelWidth;
	kernelHeight_ = kernelHeight;

	Precursor::initialize();
	createDescriptors();
	setDescriptors();
}
template<typename T>
void CudnnConvolution<T>::createDescriptors() {
	cuDNNCreateTensorDescriptor(srcTensorDescriptor_);
	cuDNNCreateFilterDescriptor(filterDescriptor_);
	cuDNNCreateConvolutionDescriptor(convolutionDescriptor_);
	cuDNNCreateTensorDescriptor(destTensorDescriptor_);
	cuDNNCreateTensorDescriptor(biasTensorDescriptor_);
}
template<typename T>
void CudnnConvolution<T>::setDescriptors() {
	cudnnDataType_t dataType = cuDNN::cuDNNInferDataType<T>();

	cuDNNSetTensorDescription(srcTensorDescriptor_,  dataType, batchSize_, sourceChannels_, sourceHeight_, sourceWidth_);
	cuDNNSetFilterDescriptor(filterDescriptor_, dataType, CUDNN_TENSOR_NCHW, destChannels_, sourceChannels_, kernelHeight_, kernelWidth_);
	cuDNNSetConvolutionDescriptor(convolutionDescriptor_, padY_, padX_, strideY_, strideX_, 1, 1, CUDNN_CROSS_CORRELATION, dataType);
	int outBatch, outChannels, outHeight, outWidth;
	cuDNNGetConvolutionNdForwardOutputDim(convolutionDescriptor_, srcTensorDescriptor_, filterDescriptor_,
			outBatch, outChannels, outHeight, outWidth);
	require_eq(outBatch, batchSize_);
	require_eq(outChannels, destChannels_);
	require_eq(outHeight, destHeight_);
	require_eq(outWidth, destWidth_);
	cuDNNSetTensorDescription(destTensorDescriptor_, dataType, batchSize_, destChannels_, destHeight_, destWidth_);
	cuDNNSetTensorDescription(biasTensorDescriptor_, dataType, 1, destChannels_, 1, 1);

	cuDNNGetConvolutionForwardAlgorithm(CudnnDataStructure::cudnnHandle_, srcTensorDescriptor_, filterDescriptor_,
			convolutionDescriptor_, destTensorDescriptor_, CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, fwdAlgorithm_);
	//std::cout<<"Forward Algorithm:"<<fwdAlgorithm_<<std::endl;


	cuDNNGetConvolutionBackwardDataAlgorithm(CudnnDataStructure::cudnnHandle_, filterDescriptor_, destTensorDescriptor_,
			convolutionDescriptor_, srcTensorDescriptor_, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, bwdDataAlgorithm_);
	//std::cout<<"Bakward Data Algorithm:"<<bwdDataAlgorithm_<<std::endl;

	cuDNNGetConvolutionBackwardFilterAlgorithm(CudnnDataStructure::cudnnHandle_, srcTensorDescriptor_, destTensorDescriptor_, convolutionDescriptor_,
			filterDescriptor_, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, bwdFilterAlgorithm_);
	//std::cout<<"Backward Filter Algorithm:"<<bwdFilterAlgorithm_<<std::endl;
}
template<typename T>
void CudnnConvolution<T>::updateBatchSize(int batchSize) {
	batchSize_ = batchSize;
	setDescriptors();
}

template<typename T>
void CudnnConvolution<T>::convolveForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source, const CudaMatrix<T> &kernel) {
	T alpha = 1;
	T beta  = 0;
	cudnnStatus_t cudnnStatus = cudnnConvolutionForward(CudnnDataStructure::cudnnHandle_, &alpha, srcTensorDescriptor_,
			source.d_elem_, filterDescriptor_, kernel.d_elem_, convolutionDescriptor_,
			fwdAlgorithm_, NULL, 0, &beta, destTensorDescriptor_, dest.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do Forward Convolution! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}
template<typename T>
void CudnnConvolution<T>::convolveBackwardData(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &errorSignalOut, const CudaMatrix<T> &filter) {
	T alpha = 1;
	T beta = 0;
	cudnnStatus_t cudnnStatus = cudnnConvolutionBackwardData(CudnnDataStructure::cudnnHandle_,
			&alpha, filterDescriptor_, filter.d_elem_, destTensorDescriptor_, errorSignalOut.d_elem_,
			convolutionDescriptor_, bwdDataAlgorithm_, NULL, 0, &beta, srcTensorDescriptor_, errorSignalIn.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do Backward Convolution w.r.t Data! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}
template<typename T>
void CudnnConvolution<T>::convolveBackwardFilter(CudaMatrix<T> &dFilter, const CudaMatrix<T> &activationIn, const CudaMatrix<T> &errorSignalOut) {
	T alpha = 1;
	T beta = 1;
	cudnnStatus_t cudnnStatus = cudnnConvolutionBackwardFilter(CudnnDataStructure::cudnnHandle_,
			&alpha, srcTensorDescriptor_, activationIn.d_elem_, destTensorDescriptor_, errorSignalOut.d_elem_,
			convolutionDescriptor_, bwdFilterAlgorithm_, NULL, 0, &beta, filterDescriptor_, dFilter.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do Backward Convolution w.r.t Filter! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}
template<typename T>
void CudnnConvolution<T>::addBias(T *data, const T *bias) {
	T alpha = 1;
	T beta = 1;

	cudnnStatus_t cudnnStatus = cudnnAddTensor(CudnnDataStructure::cudnnHandle_, &alpha, biasTensorDescriptor_, bias, &beta, destTensorDescriptor_, data);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to add Bias! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}


}


} //namespace Math



#endif /* MATH_CUDNNCONVOLUTION_HH_ */
