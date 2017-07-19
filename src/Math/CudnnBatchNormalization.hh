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
 * CudnnBatchNormalization.hh
 *
 *  Created on: Sep 9, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNBATCHNORMALIZATION_HH_
#define MATH_CUDNNBATCHNORMALIZATION_HH_
#include "CudnnDataStructure.hh"

namespace Math
{

namespace cuDNN
{

enum BNType {
	Spatial,
	PerActivation
};

template<typename T>
class CudnnBatchNormalization: public CudnnDataStructure
{
private:
	typedef CudnnDataStructure Precursor;
	cudnnTensorDescriptor_t srcTensorDescriptor_;
	cudnnTensorDescriptor_t destTensorDescriptor_;
	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_;

	BNType bnType_;

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
	CudnnBatchNormalization(BNType bnType);
	~CudnnBatchNormalization() {}

	void init(int batchSize, int sourceChannels, int sourceHeight, int sourceWidth, int destChannels,
			int destHeight, int destWidth);
	void updateBatchSize(int batchSize);

	void batchNormalizationForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source,
			const CudaVector<T> &bnGamma, const CudaVector<T> &bnBeta, double exponentialAverageFactor,
			CudaVector<T> &resultRunningMean, CudaVector<T> &resultRunningInvVariance, CudaVector<T> &resultSaveMean,
			CudaVector<T> &resultSaveVariance);
	void batchNormalizationForwardInference(CudaMatrix<T> &dest, const CudaMatrix<T> &source,
			const CudaVector<T> &bnGamma, const CudaVector<T> &bnBeta, const CudaVector<T> &estimatedMean,
			const CudaVector<T> &estimatedVariance);
	void batchNormalizationBackward(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &errorSignalOut,
			const CudaMatrix<T> &activationIn, const CudaVector<T> &bnGamma, CudaVector<T> &bnGammaDer,
			CudaVector<T> &bnBetaDer, const CudaVector<T> &savedMean, const CudaVector<T> &savedInvVariance);
};
template<typename T>
CudnnBatchNormalization<T>::CudnnBatchNormalization(BNType bnType)
	: bnType_(bnType),
	  batchSize_(0),
	  sourceWidth_(0),
	  sourceHeight_(0),
	  sourceChannels_(0),
	  destWidth_(0),
	  destHeight_(0),
	  destChannels_(0){

}
template<typename T>
void CudnnBatchNormalization<T>::init(int batchSize, int sourceChannels, int sourceHeight, int sourceWidth, int destChannels,
		int destHeight, int destWidth) {
	require_eq(sourceChannels, destChannels);
	require_eq(sourceHeight, destHeight);
	require_eq(sourceWidth, destWidth);

	batchSize_ = batchSize;
	sourceChannels_ = destChannels_ = sourceChannels;
	sourceHeight_ = destHeight_ = destHeight;
	sourceWidth_ = destWidth_ = destWidth;

	Precursor::initialize();
	createDescriptors();
	setDescriptors();
}
template<typename T>
void CudnnBatchNormalization<T>::updateBatchSize(int batchSize) {
	batchSize_ = batchSize;
	setDescriptors();
}
template<typename T>
void CudnnBatchNormalization<T>::createDescriptors() {
	cuDNNCreateTensorDescriptor(srcTensorDescriptor_);
	cuDNNCreateTensorDescriptor(destTensorDescriptor_);
	cuDNNCreateTensorDescriptor(bnScaleBiasMeanVarDesc_);
}

template<typename T>
void CudnnBatchNormalization<T>::setDescriptors() {
	cudnnDataType_t dataType = cuDNN::cuDNNInferDataType<T>();
	cuDNNSetTensorDescription(srcTensorDescriptor_, dataType, batchSize_, sourceChannels_, sourceHeight_, sourceWidth_);
	cuDNNSetTensorDescription(destTensorDescriptor_, dataType, batchSize_, destChannels_, destHeight_, destWidth_);
	cuDNNSetTensorDescription(bnScaleBiasMeanVarDesc_, dataType, 1, destChannels_, 1, 1);
}

template<typename T>
void CudnnBatchNormalization<T>::batchNormalizationForward(CudaMatrix<T> &dest, const CudaMatrix<T> &source,
		const CudaVector<T> &bnGamma, const CudaVector<T> &bnBeta, double exponentialAverageFactor,
		CudaVector<T> &resultRunningMean, CudaVector<T> &resultRunningInvVariance, CudaVector<T> &resultSaveMean,
		CudaVector<T> &resultSaveVariance){
	T alpha = 1;
	T beta = 0;
	cudnnStatus_t cudnnStatus = cudnnBatchNormalizationForwardTraining( CudnnDataStructure::cudnnHandle_, ((bnType_ == Spatial) ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION),
			&alpha, &beta, srcTensorDescriptor_, source.d_elem_, destTensorDescriptor_, dest.d_elem_, bnScaleBiasMeanVarDesc_,
			bnGamma.d_elem_, bnBeta.d_elem_, exponentialAverageFactor, resultRunningMean.d_elem_,
			resultRunningInvVariance.d_elem_, CUDNN_BN_MIN_EPSILON,	resultSaveMean.d_elem_, resultSaveVariance.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do forward BatchNormalization! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}

template<typename T>
void CudnnBatchNormalization<T>::batchNormalizationForwardInference(CudaMatrix<T> &dest, const CudaMatrix<T> &source,
		const CudaVector<T> &bnGamma, const CudaVector<T> &bnBeta, const CudaVector<T> &estimatedMean,
		const CudaVector<T> &estimatedVariance) {
	T alpha = 1;
	T beta = 0;
	cudnnStatus_t cudnnStatus = cudnnBatchNormalizationForwardInference(CudnnDataStructure::cudnnHandle_, ((bnType_ == Spatial) ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION),
			&alpha, &beta, srcTensorDescriptor_, source.d_elem_, destTensorDescriptor_, dest.d_elem_, bnScaleBiasMeanVarDesc_,
			bnGamma.d_elem_, bnBeta.d_elem_, estimatedMean.d_elem_, estimatedVariance.d_elem_, CUDNN_BN_MIN_EPSILON);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
		std::cerr<<"Unable to do forward inference BatchNormalization! cuDNNStatus:"<<cudnnStatus<<std::endl;
		exit(1);
	}
}

template<typename T>
void CudnnBatchNormalization<T>::batchNormalizationBackward(CudaMatrix<T> &errorSignalIn, const CudaMatrix<T> &errorSignalOut,
		const CudaMatrix<T> &activationIn, const CudaVector<T> &bnGamma, CudaVector<T> &bnGammaDer,
		CudaVector<T> &bnBetaDer, const CudaVector<T> &savedMean, const CudaVector<T> &savedInvVariance) {
	T alpha = 1;
	T beta = 0;

	cudnnStatus_t cudnnStatus = cudnnBatchNormalizationBackward( CudnnDataStructure::cudnnHandle_, ((bnType_ == Spatial) ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION),
			&alpha, &beta, &alpha, &beta, srcTensorDescriptor_, activationIn.d_elem_, destTensorDescriptor_, errorSignalOut.d_elem_,
			srcTensorDescriptor_, errorSignalIn.d_elem_, bnScaleBiasMeanVarDesc_, bnGamma.d_elem_, bnGammaDer.d_elem_, bnBetaDer.d_elem_,
			CUDNN_BN_MIN_EPSILON, savedMean.d_elem_, savedInvVariance.d_elem_);
	if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
			std::cerr<<"Unable to do backward BatchNormalization! cuDNNStatus:"<<cudnnStatus<<std::endl;
			exit(1);
	}
}

}

}




#endif /* MATH_CUDNNBATCHNORMALIZATION_HH_ */
