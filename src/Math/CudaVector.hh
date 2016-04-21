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
#ifndef MATH_CUDAVECTOR_HH_
#define MATH_CUDAVECTOR_HH_

#include <stdlib.h>
#include <iostream>
#include <Math/Vector.hh>
#include <Math/CudaDataStructure.hh>
#include <algorithm>
#include <Math/CudaMatrixKernelsWrapper.hh>

namespace Math {

template<typename T>
class CudaMatrix;

/*
 *
 * check maximum vector dimensions (needs to work with Cuda)
 *
 */

/*
 * CudaVector
 *
 * Vector class that makes use of GPU parallelization when compile with MODULE_CUDA and GPU is available.
 * Derives from Vector.
 * Design analogous to CudaMatrix
 */


template<typename T>
class CudaVector : protected Vector<T>, public CudaDataStructure {
	typedef Vector<T> Precursor;
	friend class CudaVector<f32>;
	friend class CudaVector<f64>;
	friend class CudaMatrix<T>;
	friend class CudaMatrix<f32>;
	friend class CudaMatrix<f64>;
protected:
	using Precursor::nRows_;
	using Precursor::nAllocatedCells_;
	using Precursor::elem_;
	using CudaDataStructure::cublasHandle;
	using CudaDataStructure::gpuMode_;
protected:
	mutable bool isComputing_;
	T *d_elem_;
public:
	// constructor with memory allocation
	CudaVector(u32 nRows = 0);

	// copy constructor
	CudaVector(const CudaVector<T> &vector);

	virtual ~CudaVector();
private:
	bool allocateGpuMemory();

public:
	// file IO
	void write(const std::string& filename);
	void read(const std::string& filename);

public:
	// resize & allocate
	// side effect: after resize content may be meaningless (if resized to a larger size)
	virtual void resize(u32 newSize);

	// resize at most to a size that has been used before
	// -> no new memory allocation, old content remains valid
	virtual void safeResize(u32 nRows);

	void resize(u32 newSize, T value) { resize(newSize); fill(value); }
	void clear();
	u32 nRows() const { return Precursor::nRows(); }
	u32 nColumns() const { return 1; }
	u32 size() const { return nRows(); }
	bool empty() const { return Precursor::empty(); }
	T& operator() (u32 index);
	T& operator[] (u32 index);
	const T& operator() (u32 index) const;
	const T& operator[] (u32 index) const;
	T& at(u32 index);
	const T& at(u32 index) const;
	// get (synchronized) value of element at position index
	const T get(u32 index) const;
public:		// iterators
typedef T* iterator;
typedef const T* const_iterator;
iterator begin();
const_iterator begin() const;
iterator end();
const_iterator end() const;
public:
// memory copy
template<typename S> void copy(const CudaVector<S>& vector);

template<typename S>
void copy(const Vector<S> &vector);

// resize to size of x & allocate
// side effect: after resize content is meaningless
void copyStructure(const CudaVector<T> &x);

// copy block from vector to specific position
void copyBlockFromVector(const Math::Vector<T> &X, u32 indexX, u32 thisIndex, u32 nElements);

// copy block from vector to specific position
void copyBlockFromVector(const Math::CudaVector<T> &X, u32 indexX, u32 thisIndex, u32 nElements);

bool isFinite() const;
public:

// addition of a vector (scaling of the vector possible)
template <typename S>
void add(const CudaVector<S> &vector, S scale = 1);

// add a constant to each element of the vector
void addConstantElementwise(T c);

// scaling of the vector
void scale(T value);

T sumOfSquares() const { return dot(*this); }

// vector dot product (result = this^T * v)
T dot(const CudaVector<T>& vector) const;

// compute distance between each column of A and v and store the result in *this
void columnwiseSquaredEuclideanDistance(const CudaMatrix<T>& A, const CudaVector<T>& v);

// matrix vector product
// this := alpha * A * X + beta * this,   or   this := alpha * A**T * X + beta * this
void multiply(const CudaMatrix<T> &A, const CudaVector<T> &x,
		bool transposed = false, T alpha = 1.0, T beta = 0.0, u32 lda = 0) const;

// set i-th component of vector to inner product of i-th column of A and i-th column of B
void columnwiseInnerProduct(const CudaMatrix<T>& A, const CudaMatrix<T>& B);

// multiply corresponding elements (this = this .* v)
void elementwiseMultiplication(const CudaVector<T>& v);

// divide corresponding elements (this = this ./ v)
void elementwiseDivision(const CudaVector<T>& v);

// division by a constant
void divide(T value);

// set all elements to zero
void setToZero();

// set all elements to value
void fill(T value);

// set all values < threshold to threshold
void ensureMinimalValue(const T threshold);

void rpropUpdate(const CudaVector<T> &newGradients, CudaVector<T> &oldGradients, CudaVector<T> &updateValues,
		const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue);

// index of minimal absolute value
u32 argAbsMin() const;

// index of maximal absolute value
u32 argAbsMax() const;

// index of maximal value
u32 argMax() const;

// elementwise exp
void exp();

// apply power function to the absolute value of each element of vector, keep original sign of each value
void signedPow(T p);

// elementwise log
void log();

// absolute value
void abs();

public:
// l1-norm of vector
T asum() const;
// just an alias
T l1norm() const;

// return sum over all elements
T sum() const;

// *this = (*this) + scale * matrixColumnSum
void addSummedRows(const CudaMatrix<T>& matrix, const T scale = 1.0);

// slightly faster version of addSummedRows that uses intermediate storage
void addSummedRows(const CudaMatrix<T>& matrix, CudaMatrix<T> &tmp, const T scale = 1.0);

// *this = (*this) + scale * matrixRowSum
void addSummedColumns(const CudaMatrix<T>& matrix, const T scale = 1.0);

// like addSummedColumns, but squares each matrix entry before summation
void addSquaredSummedColumns(const CudaMatrix<T>& matrix, const T scale = 1.0);

// this = maximum of each column in X
void getMaxOfColumns(const CudaMatrix<T> &X);

// slightly faster version of getMaxOfColumns that uses intermediate storage
void getMaxOfColumns(const CudaMatrix<T> &X, CudaMatrix<T> &tmp);

// euclidean norm => ?nrm2 s, d, sc, dz Vector 2-norm (Euclidean norm) a normal
T normEuclidean() const;

// need assignment operator, because we have a copy constructor
// pass by value ! (needed for temporary object creation)
CudaVector<T> & operator=(CudaVector<T> X);

void swap(CudaVector<T> &x);

void swap(CudaMatrix<T> &x);

public: // GPU handling
void initComputation(bool sync = true) const;
void finishComputation(bool sync = true) const;
bool isComputing() const { return isComputing_; }
public:
void show() const;
void syncAndShow() const;
};

// constructors

template<typename T>
CudaVector<T>::CudaVector(u32 nRows)
: Precursor(nRows),
  CudaDataStructure(),
  isComputing_(false),
  d_elem_(0)
  {
	allocateGpuMemory();
  }

template<typename T>
CudaVector<T>::CudaVector(const CudaVector<T> &vector)
: Precursor(vector),
  CudaDataStructure(vector),
  isComputing_(false),
  d_elem_(0)
  {
	require(!isComputing_);
	allocateGpuMemory();
  }

template<typename T>
bool CudaVector<T>::allocateGpuMemory(){
	int result = 0;
	if (gpuMode_) {
		if (d_elem_) {
			result = Cuda::free(d_elem_);
			require_eq(result, 0);
			d_elem_ = 0;
		}
		if (nRows_ > 0) {
			result = Cuda::alloc(d_elem_, nRows_);
			require_eq(result, 0);
		}
		if ((d_elem_ == 0) && (nRows_ > 0)) {
			std::cerr << "GPU: Failed to allocate memory." << std::endl;
			exit(1);
		}
	}
	return true;
}

template<typename T>
CudaVector<T>::~CudaVector(){
	if (gpuMode_){
		if (d_elem_)
			Cuda::free(d_elem_);
	}
}

template<typename T>
void CudaVector<T>::resize(u32 newSize) {
	// only reallocate memory if the size increased
	// (memory allocations slow down GPU computations if done too often... for whatever reason...)
	bool reallocate = newSize > nAllocatedCells_;
	Precursor::resize(newSize);
	if (reallocate){
		allocateGpuMemory();
	}
}

template<typename T>
void CudaVector<T>::safeResize(u32 nRows) {
	require_le(nRows, nAllocatedCells_);
	resize(nRows);
}

template<typename T>
void CudaVector<T>::clear() {
	if (gpuMode_ && d_elem_){
		Cuda::free(d_elem_);
		d_elem_ = 0;
	}
	Precursor::clear();
}

template<typename T>
T& CudaVector<T>::operator() (u32 index) {
	require(!isComputing_);
	return elem_[index];
}

template<typename T>
T& CudaVector<T>::operator[] (u32 index) {
	require(!isComputing_);
	return (*this)(index);
}

template<typename T>
const T& CudaVector<T>::operator() (u32 index) const {
	require(!isComputing_);
	return elem_[index];
}

template<typename T>
const T& CudaVector<T>::operator[] (u32 index) const {
	require(!isComputing_);
	return (*this)(index);
}


template<typename T>
T& CudaVector<T>::at(u32 index) {
	require(!isComputing_);
	return Precursor::at(index);
}

template<typename T>
const T& CudaVector<T>::at(u32 index) const {
	require(!isComputing_);
	return Precursor::at(index);
}

template<typename T>
const T CudaVector<T>::get(u32 index) const {
	if (gpuMode_ && isComputing_) {
		T val;
		int result = Cuda::copyFromGpu(&val, d_elem_ + index, 1);
		require_eq(result, 0);
		return val;
	} else {
		return Precursor::get(index);
	}
}

template<typename T>
T* CudaVector<T>::begin() {
	require(!isComputing_);
	return elem_;
}

template<typename T>
const T* CudaVector<T>::begin() const {
	require(!isComputing_);
	return elem_;
}

template<typename T>
T* CudaVector<T>::end() {
	require(!isComputing_);
	return &elem_[nRows_];
}

template<typename T>
const T* CudaVector<T>::end() const {
	require(!isComputing_);
	return &elem_[nRows_];
}

// TODO CUDA implementation works only for identical types !!
template<typename T>
template<typename S>
void CudaVector<T>::copy(const Math::CudaVector<S> & x) {
	require(isComputing_);
	require(x.isComputing_);
	if (gpuMode_){
		require_eq(x.nRows(), nRows_);
		int result = Cuda::copy(cublasHandle, nRows_, x.d_elem_, 1, d_elem_, 1);
		require_eq(result, 0);
	}
	else
		Precursor::copy(x);
}

template<typename T>
template<typename S>
void CudaVector<T>::copy(const Vector<S> &vector){
	require(!isComputing_);
	Precursor::copy(vector);
}

template<typename T>
void CudaVector<T>::copyStructure(const Math::CudaVector<T> & x) {
	if (x.nRows_ != nRows_)
		resize(x.nRows_);
}

template<typename T>
void CudaVector<T>::copyBlockFromVector(const Math::Vector<T> &X, u32 indexX, u32 thisIndex, u32 nElements) {
	require(!isComputing_);
	Precursor::copyBlockFromVector(X, indexX, thisIndex, nElements);
}

template<typename T>
void CudaVector<T>::copyBlockFromVector(const Math::CudaVector<T> &X, u32 indexX, u32 thisIndex, u32 nElements) {
	require(isComputing_);
	require(X.isComputing_);
	require_le(thisIndex + nElements, nRows_);
	require_le(indexX + nElements, X.nRows_);
	if (gpuMode_){
		const T *posX = X.d_elem_ + indexX;
		T * posThis = d_elem_ + thisIndex;
		Cuda::copy(cublasHandle, nElements, posX, 1, posThis, 1);
	}
	else {
		Precursor::copyBlockFromVector(X, indexX, thisIndex, nElements);
	}
}

template<typename T>
bool CudaVector<T>::isFinite() const {
	require(!isComputing_);
	return Precursor::isFinite();
}

// ----------------------------------------------------------------------------
//		Math operations
// ----------------------------------------------------------------------------

template<typename T>
template<typename S>
void CudaVector<T>::add(const CudaVector<S>& vector, S scale) {
	require(isComputing_);
	require(vector.isComputing_);
	if (gpuMode_) {
		require_eq(nRows_, vector.nRows());
		int result = Cuda::axpy(cublasHandle, nRows_, scale, vector.d_elem_,1,  d_elem_, 1);
		require_eq(result, 0);
	} else {
		Precursor::add(vector, scale);
	}
}

template<typename T>
void CudaVector<T>::addConstantElementwise(T c) {
	require(isComputing_);
	if (gpuMode_)
		Cuda::addConstantElementwise(c, d_elem_, nRows_, 1);
	else
		Precursor::addConstantElementwise(c);
}

template<typename T>
void CudaVector<T>::scale(T scale) {
	require(isComputing_);
	if (gpuMode_) {
		int result = Cuda::scal(cublasHandle, nRows_, scale, d_elem_, 1);
		require_eq(result, 0);
	} else {
		Precursor::scale(scale);
	}
}

template<typename T>
T CudaVector<T>::dot(const CudaVector<T>& vector) const {
	require(isComputing_);
	require(vector.isComputing_);
	if (gpuMode_){
		T dotProduct = 0;
		int result = Cuda::dot(cublasHandle, nRows_, vector.d_elem_, 1, d_elem_, 1, dotProduct);
		require_eq(result, 0);
		return dotProduct;
	} else {
		return Precursor::dot(vector);
	}
}

template<typename T>
void CudaVector<T>::columnwiseSquaredEuclideanDistance(const CudaMatrix<T>& A, const CudaVector<T>& v) {
	require(isComputing_);
	require(A.isComputing_);
	require(v.isComputing_);
	if (gpuMode_) {
		require_eq(nRows_, A.nColumns());
		require_eq(A.nRows(), v.nRows());
		setToZero();
		_cuda_columnwiseSquaredEuclideanDistance(A.d_elem_, A.nRows(), A.nColumns(), v.d_elem_, d_elem_);
	} else {
		Precursor::columnwiseSquaredEuclideanDistance(A, v);
	}
}

template<typename T>
void CudaVector<T>::multiply(const CudaMatrix<T> &A, const CudaVector<T> &x, bool transposed, T alpha, T beta, u32 lda) const {
	require(isComputing_);
	if (gpuMode_) {
		require_le(lda,A.nRows());
		if (lda == 0)
			lda = A.nRows();
		if (!transposed && lda == A.nRows()){
			require_eq(x.nRows(), A.nColumns());
			require_eq(nRows_, A.nRows());
		}
		else if (transposed && lda == A.nRows()){
			require_eq(x.nRows(), A.nRows());
			require_eq(nRows_, A.nColumns());
		}
		// TODO checks with non-default lda ?
		int result = Cuda::gemv(cublasHandle, transposed, A.nRows(), A.nColumns(), alpha, A.d_elem_, lda, x.d_elem_, 1, beta, d_elem_, 1);
		require_eq(result, 0);
	} else {
		Precursor::multiply(A, x, transposed, alpha, beta, lda);
	}
}

template<typename T>
void CudaVector<T>::columnwiseInnerProduct(const Math::CudaMatrix<T>& A, const Math::CudaMatrix<T>& B) {
	require(isComputing_);
	require(A.isComputing());
	require(B.isComputing());
	if (gpuMode_) {
		require_eq(A.nRows(), B.nRows());
		require_eq(A.nColumns(), B.nColumns());
		require_eq(nRows_, A.nColumns());
		u32 matrixRows = A.nRows();
		// TODO: for now only parallelized within the columns, implement a better parallelization
		for (u32 column = 0; column < A.nColumns(); column++) {
			T dotProduct = 0;
			int result = Cuda::dot(cublasHandle, matrixRows, A.d_elem_ + column * matrixRows, 1,
					B.d_elem_ + column * matrixRows, 1, dotProduct);
			require_eq(result, 0);
			Cuda::copyToGpu(d_elem_ + column, &dotProduct, 1);
		}
	} else {
		Precursor::columnwiseInnerProduct(A, B);
	}
}

template<typename T>
void CudaVector<T>::elementwiseMultiplication(const CudaVector<T>& v) {
	require(isComputing_);
	require(v.isComputing_);
	if (gpuMode_) {
		require_eq(nRows_, v.nRows_);
		Cuda::elementwiseMultiplication(d_elem_, v.d_elem_, v.nRows_, 1);
	} else {
		Precursor::elementwiseMultiplication(v);
	}
}

template<typename T>
void CudaVector<T>::elementwiseDivision(const CudaVector<T>& v) {
	require(isComputing_);
	require(v.isComputing_);
	if (gpuMode_) {
		require_eq(nRows_, v.nRows_);
		Cuda::elementwiseDivision(d_elem_, v.d_elem_, v.nRows_, 1);
	} else {
		Precursor::elementwiseDivision(v);
	}
}

template<typename T>
void CudaVector<T>::rpropUpdate(const CudaVector<T> &newGradients, CudaVector<T> &oldGradients, CudaVector<T> &updateValues,
		const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue) {
	require(isComputing_);
	require(newGradients.isComputing_);
	require(oldGradients.isComputing_);
	require(updateValues.isComputing_);
	require_eq(oldGradients.nRows(), nRows_);
	require_eq(newGradients.nRows(), nRows_);
	require_eq(updateValues.nRows(), nRows_);
	if (gpuMode_){
		Cuda::rpropUpdate(d_elem_, newGradients.d_elem_, oldGradients.d_elem_, updateValues.d_elem_,
				increasingFactor, decreasingFactor, maxUpdateValue, minUpdateValue, newGradients.nRows_, 1);
	}
	else
		Precursor::rpropUpdate(newGradients, oldGradients, updateValues,
				increasingFactor, decreasingFactor, maxUpdateValue, minUpdateValue);
}

template<typename T>
u32 CudaVector<T>::argAbsMin() const {
	require(isComputing_);
	if (gpuMode_) {
		int result = 0;
		Cuda::iamin(cublasHandle, nRows_, d_elem_, 1, &result);
		return result;
	} else {
		return Precursor::argAbsMin();
	}
}

template<typename T>
u32 CudaVector<T>::argAbsMax() const {
	require(isComputing_);
	if (gpuMode_) {
		int result = 0;
		Cuda::iamax(cublasHandle, nRows_, d_elem_, 1, &result);
		return result;
	} else {
		return Precursor::argAbsMax();
	}
}

template<typename T>
u32 CudaVector<T>::argMax() const {
	require(isComputing_);
	if (gpuMode_) {
		u32 result = 0;
		u32 *resultDev;
		Cuda::alloc(resultDev, 1);
		Cuda::argMax(d_elem_, nRows_, 1, resultDev);
		Cuda::copyFromGpu(&result, resultDev, 1);
		Cuda::free(resultDev);
		return result;
	} else {
		return Precursor::argMax();
	}
}

template<typename T>
void CudaVector<T>::exp() {
	require(isComputing_);
	if (gpuMode_)
		Cuda::exp(d_elem_, nRows_ , 1);
	else
		Precursor::exp();
}

template<typename T>
void CudaVector<T>::signedPow(T p) {
	require(isComputing_);
	if (gpuMode_)
		Cuda::signedPow(d_elem_, nRows_ , 1, p);
	else
		Precursor::signedPow(p);
}

template<typename T>
void CudaVector<T>::log() {
	require(isComputing_);
	if (gpuMode_)
		Cuda::log(d_elem_, nRows_ , 1);
	else
		Precursor::log();
}

template<typename T>
void CudaVector<T>::abs() {
	require(isComputing_);
	if (gpuMode_)
		Cuda::abs(d_elem_, nRows_ , 1);
	else
		Precursor::abs();
}

template<typename T>
void CudaVector<T>::divide(T value) {
	require(isComputing_);
	if (gpuMode_) {
		scale((T) 1 / value);
	} else {
		Precursor::divide(value);
	}
}

template<typename T>
void CudaVector<T>::setToZero() {
	if (gpuMode_ && isComputing_) {
		int result = Cuda::memSet(d_elem_, 0, nRows_);
		require_eq(result, 0);
	} else {
		Precursor::setToZero();
	}
}

template<typename T>
void CudaVector<T>::fill(T value) {
	require(isComputing_);
	if (gpuMode_) {
		Cuda::fill(d_elem_, value, nRows_, 1);
	} else {
		Precursor::fill(value);
	}
}

template<typename T>
void CudaVector<T>::ensureMinimalValue(const T threshold) {
	require(isComputing_);
	if (gpuMode_) {
		Cuda::ensureMinimalValue(d_elem_, threshold, nRows_, 1);
	} else {
		Precursor::ensureMinimalValue(threshold);
	}
}

template<typename T>
T CudaVector<T>::asum() const {
	require(isComputing_);
	int resultb = 0;
	if (gpuMode_) {
		T result;
		resultb = Cuda::asum(cublasHandle, nRows_, d_elem_, 1, &result);
		require_eq(resultb, 0);
		return result;
	} else {
		return Precursor::asum();
	}
}

template<typename T>
T CudaVector<T>::l1norm() const {
	return asum();
}

template<typename T>
T CudaVector<T>::sum() const {
	require(isComputing_);
	if (gpuMode_) {
		T result = 0;
		T *resultDev;
		Cuda::alloc(resultDev, 1);
		Cuda::sum(d_elem_, nRows_, 1, resultDev);
		Cuda::copyFromGpu(&result, resultDev, 1);
		Cuda::free(resultDev);
		return result;
	} else {
		return Precursor::sum();
	}
}

template<typename T>
void CudaVector<T>::addSummedColumns(const CudaMatrix<T>& matrix, const T scale) {
	require(isComputing_);
	require(matrix.isComputing());
	require_eq(matrix.nRows(), nRows_);
	if (gpuMode_) {
		Cuda::addSummedColumns(d_elem_, matrix.d_elem_, matrix.nRows_ , matrix.nColumns_, scale);
	} else {
		Precursor::addSummedColumns(matrix, scale);
	}
}

template<typename T>
void CudaVector<T>::addSquaredSummedColumns(const CudaMatrix<T>& matrix, const T scale) {
	require(isComputing_);
	require(matrix.isComputing());
	require_eq(matrix.nRows(), nRows_);
	if (gpuMode_) {
		Cuda::addSquaredSummedColumns(d_elem_, matrix.d_elem_, matrix.nRows_ , matrix.nColumns_, scale);
	} else {
		Precursor::addSquaredSummedColumns(matrix, scale);
	}
}

template<typename T>
void CudaVector<T>::addSummedRows(const CudaMatrix<T>& matrix, const T scale) {
	require(isComputing_);
	require(matrix.isComputing());
	require_eq(matrix.nColumns(), nRows_);
	if (gpuMode_) {
		Cuda::addSummedRows(d_elem_, matrix.d_elem_, matrix.nRows_ , matrix.nColumns_, scale);
	} else {
		Precursor::addSummedRows(matrix, scale);
	}
}

template<typename T>
void CudaVector<T>::addSummedRows(const CudaMatrix<T>& matrix, CudaMatrix<T> &tmp, const T scale) {
	require(isComputing_);
	require(matrix.isComputing());
	require(tmp.isComputing());
	require_eq(matrix.nColumns(), nRows_);
	require_eq(tmp.nColumns(), matrix.nColumns());
	if (gpuMode_) {
		Cuda::addSummedRows(d_elem_, matrix.d_elem_, matrix.nRows_ , matrix.nColumns_, tmp.d_elem_, tmp.nRows_, scale);
	} else {
		Precursor::addSummedRows(matrix, scale);
	}
}

template<typename T>
void CudaVector<T>::getMaxOfColumns(const CudaMatrix<T>& matrix) {
	require(isComputing_);
	require(matrix.isComputing());
	require_eq(matrix.nColumns(), nRows_);
	if (gpuMode_) {
		Cuda::getMaxOfColumns(d_elem_, matrix.d_elem_, matrix.nRows_ , matrix.nColumns_);
	} else {
		Precursor::getMaxOfColumns(matrix);
	}
}

template<typename T>
void CudaVector<T>::getMaxOfColumns(const CudaMatrix<T> &X, CudaMatrix<T> &tmp){
	require(isComputing_);
	require(X.isComputing());
	require(tmp.isComputing());
	require_eq(X.nColumns(), nRows_);
	require_eq(tmp.nColumns(), X.nColumns());
	if (gpuMode_)
		Cuda::getMaxOfColumns(d_elem_, X.d_elem_, X.nRows_ , X.nColumns_, tmp.d_elem_, tmp.nRows_);
	else
		Precursor::getMaxOfColumns(X);
}

template<typename T>
T CudaVector<T>::normEuclidean() const {
	require(isComputing_);
	if (gpuMode_) {
		T result;
		Cuda::nrm2(cublasHandle, nRows_, d_elem_, 1, &result);
		return result;
	} else {
		return Precursor::normEuclidean();
	}
}

template<typename T>
CudaVector<T> &CudaVector<T>::operator=(CudaVector<T> rhs) {
	swap(rhs);
	return *this;
}

template<typename T>
void CudaVector<T>::swap(CudaVector<T> &x) {
	require_eq(x.gpuMode_, gpuMode_);
	require_eq(x.isComputing_, isComputing_);
	Precursor::swap(x);
	std::swap(d_elem_, x.d_elem_);
}

template<typename T>
void CudaVector<T>::swap(CudaMatrix<T> &x) {
	require_eq(x.gpuMode_, gpuMode_);
	require_eq(x.isComputing_, isComputing_);
	Precursor::swap(x);
	std::swap(d_elem_, x.d_elem_);
}

// ----------------------------------------------------------------------------
//		GPU handling
// ----------------------------------------------------------------------------

template<typename T>
void CudaVector<T>::initComputation(bool sync) const {
	if (gpuMode_ && !isComputing_){
		if (sync)
			Cuda::copyToGpu(d_elem_, elem_, nRows_);
	}
	isComputing_ = true;
}

template<typename T>
void CudaVector<T>::finishComputation(bool sync) const {
	if (gpuMode_ && isComputing_) {
		if (d_elem_ && sync)
			Cuda::copyFromGpu(elem_, d_elem_, nRows_);
	}
	isComputing_ = false;
}

template<typename T>
void CudaVector<T>::show() const {
	require(!isComputing_);
	Precursor::show();
}

template<typename T>
void CudaVector<T>::syncAndShow() const {
	if (isComputing_ && gpuMode_){
		Cuda::copyFromGpu(elem_, d_elem_, nRows_);
	}
	Precursor::show();
}

template<typename T>
void CudaVector<T>::write(const std::string& filename) {
	require(!isComputing_);
	Precursor::write(filename);
}

template<typename T>
void CudaVector<T>::read(const std::string& filename) {
	require(!isComputing_);
	Precursor::read(filename);
}

} // namespace Math

#endif /* MATH_CUDAVECTOR_HH_ */
