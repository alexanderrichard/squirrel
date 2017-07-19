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
 * CudnnPooling.hh
 *
 *  Created on: Sep 6, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNPOOLING_HH_
#define MATH_CUDNNPOOLING_HH_

#include "CudnnDataStructure.hh"
#include "CudaMatrix.hh"

namespace Math
{

namespace cuDNN
{

enum PoolingType {
	MaxPooling,
	AvgPooling
};

template<typename T>
class CudnnPooling : public CudnnDataStructure {
private:
	typedef CudnnDataStructure Precursor;
	cudnnTensorDescriptor_t srcTensorDescriptor_;
	cudnnTensorDescriptor_t destTensorDescriptor_;
	cudnnPoolingDescriptor_t poolingDesc_;

	PoolingType poolingType_;
	int poolSize_;
	int stride_;
	int padX_;
	int padY_;

	int batchSize_;
	int sourceWidth_;
	int sourceHeight_;
	int sourceChannels_;
	int destWidth_;
	int destHeight_;
	int destChannels_;

	void createDescriptors();
	void setDescriptors();
public:
	CudnnPooling(PoolingType poolType);
	~CudnnPooling() {}

	void init(int poolSize, int stride, int padX, int padY,
			int batchSize, int sourceWidth, int sourceHeight, int sourceChannles,
			int destWidth, int destHeight, int destChannels);
	void updateBatchSize(int batchSize);

	void poolingForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source);
	void poolingBackward(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &activationIn,
			const CudaMatrix<T> &errorSignalOut, const CudaMatrix<T> &activationOut);
};



template<typename T>
CudnnPooling<T>::CudnnPooling(PoolingType poolType) :
	Precursor(),
	poolingType_(poolType),
	poolSize_(0),
	stride_(0),
	padX_(0),
	padY_(0),
	batchSize_(0),
	sourceWidth_(0),
	sourceHeight_(0),
	sourceChannels_(0),
	destWidth_(0),
	destHeight_(0),
	destChannels_(0) {

}
template<typename T>
void CudnnPooling<T>::createDescriptors() {
	cuDNNCreateTensorDescriptor(srcTensorDescriptor_);
	cuDNNCreatePoolingDescriptor(poolingDesc_);
	cuDNNCreateTensorDescriptor(destTensorDescriptor_);
}
template<typename T>
void CudnnPooling<T>::setDescriptors() {

	cudnnDataType_t dataType = cuDNN::cuDNNInferDataType<T>();
	cuDNNSetTensorDescription(srcTensorDescriptor_, dataType,
			batchSize_, sourceChannels_, sourceHeight_, sourceWidth_);

	cuDNNSetPoolingDescriptor(poolingDesc_,
			(poolingType_ == MaxPooling ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING),
			poolSize_, padX_, padY_, stride_);
	int outputDim[4] = {batchSize_, sourceChannels_, sourceHeight_, sourceWidth_};
	cuDNNGetPoolingNdForwardOutputDim(poolingDesc_, srcTensorDescriptor_, outputDim);

	require_eq(batchSize_, outputDim[0]);
	require_eq(destChannels_, outputDim[1]);
	require_eq(destHeight_, outputDim[2]);
	require_eq(destWidth_, outputDim[3]);

	cuDNNSetTensorDescription(destTensorDescriptor_, dataType,
			batchSize_, destChannels_, destHeight_, destWidth_);


}
template<typename T>
void CudnnPooling<T>::init(int poolSize, int stride, int padX, int padY,
		int batchSize, int sourceWidth, int sourceHeight, int sourceChannles,
		int destWidth, int destHeight, int destChannels) {
	poolSize_ = poolSize;
	stride_ = stride;
	padX_ = padX;
	padY_ = padY;
	batchSize_ = batchSize;
	sourceWidth_ = sourceWidth;
	sourceHeight_ = sourceHeight;
	sourceChannels_ = sourceChannles;
	destWidth_ = destWidth;
	destHeight_ = destHeight;
	destChannels_ = destChannels;

	require_eq(destChannels_, sourceChannels_);

	Precursor::initialize();
	createDescriptors();
	setDescriptors();
}
template<typename T>
void CudnnPooling<T>::updateBatchSize(int batchSize) {
	batchSize_ = batchSize;
	setDescriptors();
}
template<typename T>
void CudnnPooling<T>::poolingForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source) {
	T alpha = 1;
	T beta = 0;

	cudnnStatus_t cudnnStatus = cudnnPoolingForward(CudnnDataStructure::cudnnHandle_, poolingDesc_,
			&alpha, srcTensorDescriptor_, source.d_elem_, &beta, destTensorDescriptor_, dest.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do forward pooling! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}
template<typename T>
void CudnnPooling<T>::poolingBackward(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &activationIn,
		const CudaMatrix<T> &errorSignalOut, const CudaMatrix<T> &activationOut) {
	T alpha = 1;
	T beta = 0;

	cudnnStatus_t cudnnStatus = cudnnPoolingBackward(CudnnDataStructure::cudnnHandle_, poolingDesc_,
			&alpha, destTensorDescriptor_, activationOut.d_elem_, destTensorDescriptor_, errorSignalOut.d_elem_,
			srcTensorDescriptor_, activationIn.d_elem_, &beta, srcTensorDescriptor_, errorSignalIn.d_elem_);

	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do backward pooling! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}


} //namespace cuDNN

}//namespace Math



#endif /* MATH_CUDNNPOOLING_HH_ */
