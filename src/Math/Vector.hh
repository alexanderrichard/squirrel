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

#ifndef MATH_VECTOR_HH_
#define MATH_VECTOR_HH_

#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <typeinfo>

#include <Core/CommonHeaders.hh>
#include <Math/Blas.hh>
#include <Math/FastVectorOperations.hh>

#include <iostream>		/** to use std::cout */
#include <sstream>

#include <Math/Matrix.hh>

namespace Math {

template<typename T>
class Matrix;

/*
 * fast vector implementation
 * vectors are assumed to be column vectors
 * note: BLAS interface assumes int as size type, therefore size is limited by Core::max<int>()
 */

template<typename T>
class Vector {
	friend class Vector<f32>;
	friend class Vector<f64>;
	friend class Matrix<T>;
	friend class Matrix<f32>;
	friend class Matrix<f64>;
protected:
	// the number of sizeof(T) elements that are actually allocated
	// (may differ from nRows_ * nColumns_ due to lazy resize)
	u32 nAllocatedCells_;
	u32 nRows_;
	T *elem_;
public:
	Vector(u32 nRows = 0);			// constructor with memory allocation

	Vector(const Vector<T> &vector);	// (deep) copy constructor

	// destructor
	virtual ~Vector() { clear(); }

	bool allocate();

private:
	void writeBinaryHeader(Core::IOStream& stream);
	void writeAsciiHeader(Core::IOStream& stream);
	void writeBinary(Core::IOStream& stream);
	void writeAscii(Core::IOStream& stream, bool scientific);
	void readHeader(Core::IOStream& stream);
	void read(Core::IOStream& stream);

public:
	// file IO
	void write(const std::string& filename, bool scientific = false);
	void read(const std::string& filename);

public:
	u32 nRows() const { return nRows_; }
	u32 nColumns() const { return 1; }
	u32 size() const { return (*this).nRows(); }
	bool empty() const { return ((*this).size() == 0); }
	// returns whether all matrix entries are finite
	bool isFinite() const;

public:
	// resize & allocate
	// side effect: after resize content may be meaningless (if resized to a larger size)
	// if reallocate is true enforce reallocation of memory
	virtual void resize(u32 newSize, bool reallocate = false);

	// resize at most to a size that has been used before
	// -> no new memory allocation, old content remains valid
	virtual void safeResize(u32 nRows);

	void clear();
public:
	// copy
	template<typename S>
	void copy(const Vector<S>& vector);

	void copyStructure(const Vector<T> &vector);

	// copy block from vector to specific position
	void copyBlockFromVector(const Math::Vector<T> &X, u32 indexX, u32 thisIndex, u32 nElements);

public:		// iterators
	typedef T* iterator;
	typedef const T* const_iterator;
	iterator begin() { return elem_; }
	const_iterator begin() const { return elem_; }
	iterator end() { return &elem_[nRows_]; }
	const_iterator end() const { return &elem_[nRows_]; }

public:
	// addition of a vector (scaling of the vector possible)
	template<typename S>
	void add(const Vector<S> &vector, S scale = 1) {
		Math::axpy<S,T>((*this).size(), (T) scale, vector.begin(), 1, (*this).begin(), 1);
	}

	// add a constant to each element of the vector
	void addConstantElementwise(T c) { std::transform(begin(), end(), begin(), std::bind2nd(std::plus<T>(), c)); }

	// scaling of the vector
	void scale(T value) { Math::scal(nRows_, value, elem_, 1); }

	// @return sum of squared matrix entries
	T sumOfSquares() const { return dot (*this); }

	// vector dot product (result = this^T * v)
	T dot(const Vector<T>& vector) const;

	// compute distance between each column of A and v and store the result in *this
	void columnwiseSquaredEuclideanDistance(const Matrix<T>& A, const Vector<T>& v);

	// matrix vector product
	// this := alpha * A * X + beta * this,   or   this := alpha * A**T * X + beta * this
	void multiply(const Matrix<T> &A, const Vector<T> &x,
			bool transposed = false, T alpha = 1.0, T beta = 0.0, u32 lda = 0) const;

	// set i-th component of vector to inner product of i-th column of A and i-th column of B
	void columnwiseInnerProduct(const Matrix<T>& A, const Matrix<T>& B);

	// multiply corresponding elements (this = this .* v)
	void elementwiseMultiplication(const Vector<T>& v);

	// divide corresponding elements (this = this ./ v)
	void elementwiseDivision(const Vector<T>& v);

	// division by a constant
	void divide(T value) { scale((T) 1 / value); }

	// set all elements to a constant value (e.g. zero)
	void setToZero();
	void fill(T value);

	// set all values < threshold to threshold
	void ensureMinimalValue(const T threshold);

	void rpropUpdate(const Vector<T> &newGradients, Vector<T> &oldGradients, Vector<T> &updateValues, const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue);

	// index of minimal absolute value
	u32 argAbsMin() const;

	// index of maximal absolute value
	u32 argAbsMax() const;

	// index of maximal value
	u32 argMax() const;

	// return maximal value in vector
	T max() const;

	// elementwise exp
	void exp();

	// apply power function to the absolute value of each element of vector, keep original sign of each value
	void signedPow(T p);

	// elementwise log
	void log();

	// absolute values
	void abs();

public:
	Vector<T> &operator=(Vector<T> rhs);
	// vector access
	T& operator() (u32 index) { return elem_[index]; }
	T& operator[] (u32 index) { return (*this)(index); }
	T& operator() (u32 index) const { return elem_[index]; }
	T& operator[] (u32 index) const { return (*this)(index); }
	T& at(u32 index);
	const T& at(u32 index) const;
	const T get(u32 index) const;
public:
	// l1-norm of vector
	T asum() const { return Math::asum((*this).size(), (*this).begin(), 1); }
	// just an alias
	T l1norm() const { return asum(); }

	// return sum over all elements
	T sum() const;

	// *this = (*this) + scale * matrixRowSum
	void addSummedColumns(const Matrix<T>& matrix, const T scale = 1.0);

	// *this = (*this) + scale * (matrixChannelSum)
	void addSummedColumnsChannelWise(const Matrix<T>& matrix, const u32 channels, const T scale = 1.0);

	// like addSummedColumns, but squares each matrix entry before summation
	void addSquaredSummedColumns(const Matrix<T>& matrix, const T scale = 1.0);

	// *this = (*this) + scale * matrixColumnSum
	void addSummedRows(const Matrix<T>& matrix, const T scale = 1.0);

	void getMaxOfColumns(const Matrix<T> &X);

	// euclidean norm => ?nrm2 s, d, sc, dz Vector 2-norm (Euclidean norm) a normal
	T normEuclidean() const { return Math::nrm2((*this).size(), (*this).begin(), 1); }

	// chi-square distance \sum_d (x_d - y_d)^2 / (x_d + y_d)
	T chiSquareDistance(const Vector<T>& v) const;

	// modified chi-square distance \sum_d 2 * x_d * y_d / (x_d + y_d)
	// see Efficient Additive Kernels via Explicit Feature Maps, Vedaldi and Zisserman
	T modifiedChiSquareDistance(const Vector<T>& v) const;

	// compute the chi square feature map of the vector v
	// see Efficient Additive Kernels via Explicit Feature Maps, Vedaldi and Zisserman
	void chiSquareFeatureMap(const Vector<T>& v, u32 nSamples, T samplingDistance);

	// swap two vectors
	void swap(Vector<T>& vector);

	// swap matrix and vector (matrix will end up being a single column, former matrix columns are concatenated in the vector)
	void swap(Matrix<T>& X);

	std::string toString(bool transpose = false) const;
};


template<typename T>
bool Vector<T>::allocate() {
	if (elem_)
		delete [] elem_;
	elem_ = nRows_ > 0 ? new T[nRows_] : 0;
	nAllocatedCells_ = nRows_ > 0 ? nRows_ : 0;
	return true;
}

// ----------------------------------------------------------------------------
//		Vector constructors
// ----------------------------------------------------------------------------
template<typename T>
Vector<T>::Vector(u32 size) :
	nAllocatedCells_(0),
	nRows_(size),
	elem_(0)
{
	allocate();
}

/**	Copy constructor
 */
template<typename T>
Vector<T>::Vector(const Vector<T> &vector) :
	nAllocatedCells_(0),
	nRows_(vector.nRows_),
	elem_(0)
{
	allocate();
	// copy all elements of the vector
	copy(vector);
}

// ----------------------------------------------------------------------------
//		Vector resize
// ----------------------------------------------------------------------------
template<typename T>
void Vector<T>::resize(u32 newSize, bool reallocate) {
	reallocate |= newSize > nAllocatedCells_;
	nRows_ = newSize;
	if (reallocate)
		allocate();
}

template<typename T>
void Vector<T>::safeResize(u32 nRows) {
	require_le(nRows, nAllocatedCells_);
	resize(nRows, false);
}

template<typename T>
void Vector<T>::clear() {
	if (elem_)
		delete [] elem_;
	elem_ = 0;
	nRows_ = 0;
	nAllocatedCells_ = 0;
}

/**	Copy the vector.
 *
 *	Create a copy of the vector by:
 *	- cast each element (different type)
 *	- copy the memory (same type) .
 */
template<typename T>
template<typename S>
void Vector<T>::copy(const Vector<S>& vector) {
	// check the dimension
	require(nRows_ == vector.nRows());
	// copy memory (use copy from Math::Blas.hh)
	Math::copy<S,T>(nRows_, vector.elem_, 1, elem_, 1);
}

template<typename T>
void Vector<T>::copyStructure(const Vector<T>& vector) {
	resize(vector.nRows());
}

template<typename T>
void Vector<T>::copyBlockFromVector(const Math::Vector<T> &X, u32 indexX, u32 thisIndex, u32 nElements) {
	require_le(thisIndex + nElements, nRows_);
	require_le(indexX + nElements, X.nRows_);
	const T *posX =  &X.at(indexX);
	T * posThis = &at(thisIndex);
	Math::copy(nElements, posX, 1, posThis, 1);
}

template<typename T>
T& Vector<T>::at(u32 index) {
	require(index < nRows_);
	return (*this)(index);
}

template<typename T>
const T& Vector<T>::at(u32 index) const {
	require(index < nRows_);
	return elem_[index];
}

template<typename T>
const T Vector<T>::get(u32 index) const {
	require(index < nRows_);
	return at(index);
}

template<typename T>
bool Vector<T>::isFinite() const {
	for (u32 index = 0; index < nRows_; ++index) {
		T val = at(index);
		if (Types::isNan(val) || val > Types::max<T>() || val < Types::min<T>())
			return false;
	}
	return true;
}

template<typename T>
T Vector<T>::dot(const Math::Vector<T> &v) const {
	require(nRows_ == v.nRows());
	return Math::dot(nRows_, elem_, 1, v.elem_, 1);
}

template<typename T>
void Vector<T>::columnwiseSquaredEuclideanDistance(const Matrix<T>& A, const Vector<T>& v) {
	require(!A.needsU64Space_);
	require_eq(nRows_, A.nColumns());
	require_eq(A.nRows(), v.nRows());
	setToZero();
#pragma omp parallel for
	for (u32 i = 0; i < A.nRows(); i++) {
		for (u32 j = 0; j < A.nColumns(); j++) {
			at(j) += (A.at(i,j) - v.at(i)) * (A.at(i,j) - v.at(i));
		}
	}
}

template<typename T>
void Vector<T>::multiply(const Math::Matrix<T> &A, const Math::Vector<T> &x, bool transposed, T alpha, T beta, u32 lda) const {
	require(!A.needsU64Space_);
	require_le(lda, A.nRows());
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
	CBLAS_TRANSPOSE tr = transposed ? CblasTrans : CblasNoTrans;
	// assume col major order
	Math::gemv<T>(CblasColMajor, tr, A.nRows(), A.nColumns(), alpha, A.begin(), lda, x.elem_, 1, beta, elem_, 1);
}

template<typename T>
void Vector<T>::columnwiseInnerProduct(const Math::Matrix<T>& A, const Math::Matrix<T>& B) {
	require(!A.needsU64Space_);
	require(!B.needsU64Space_);
	require_eq(A.nRows(), B.nRows());
	require_eq(A.nColumns(), B.nColumns());
	require_eq(nRows_, A.nColumns());
	u32 matrixRows = A.nRows();
	// TODO: for now only parallelized within the columns, implement a better parallelization
	for (u32 column = 0; column < A.nColumns(); column++) {
		at(column) = Math::dot(matrixRows, A.begin() + column * matrixRows, 1, B.begin() + column * matrixRows, 1);
	}
}

template<typename T>
void Vector<T>::elementwiseMultiplication(const Vector<T>& v) {
	require(nRows_ == v.nRows());
	std::transform(begin(), end(), v.begin(), begin(), std::multiplies<T>());
}

template<typename T>
void Vector<T>::elementwiseDivision(const Vector<T>& v) {
	require(nRows_ == v.nRows());
	std::transform(begin(), end(), v.begin(), begin(), std::divides<T>());
}

template<typename T>
void Vector<T>::rpropUpdate(const Vector<T> &newGradients, Vector<T> &oldGradients, Vector<T> &updateValues,
		const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue) {
	require_eq(oldGradients.nRows(), nRows_);
	require_eq(newGradients.nRows(), nRows_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++) {
		T change = oldGradients.at(i) *  newGradients.at(i);
		if (change > 0) {
			updateValues.at(i) *= increasingFactor;
			if (updateValues.at(i) > maxUpdateValue)
				updateValues.at(i) = maxUpdateValue;
		} else if (change < 0) {
			updateValues.at(i) *= decreasingFactor;
			if (updateValues.at(i) < minUpdateValue)
				updateValues.at(i) = minUpdateValue;
		}
		if (newGradients.at(i) > 0)
			at(i) += -updateValues.at(i);
		else if (newGradients.at(i) < 0)
			at(i) += updateValues.at(i);
		oldGradients.at(i) = newGradients.at(i);
	}
}

template<typename T>
u32 Vector<T>::argAbsMin() const {
	return Math::iamin(nRows_, elem_, 1);
}

template<typename T>
u32 Vector<T>::argAbsMax() const {
	return Math::iamax(nRows_, elem_, 1);
}

template<typename T>
u32 Vector<T>::argMax() const {
	T max = Types::min<T>();
	u32 argMax = 0;
	for (u32 i = 0; i < nRows_; i++) {
		argMax = (at(i) > max ? i : argMax);
		max = (at(i) > max ? at(i) : max);
	}
	return argMax;
}

template<typename T>
T Vector<T>::max() const {
	T max = Types::min<T>();
	for (u32 i = 0; i < nRows_; i++)
		max = (at(i) > max ? at(i) : max);
	return max;
}

template<typename T>
void Vector<T>::exp() {
	vr_exp(nRows_, elem_ , elem_);
}

template<typename T>
void Vector<T>::signedPow(T p) {
# pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++) {
		if (elem_[i] < 0)
			elem_[i] = -std::pow(-elem_[i], p);
		else
			elem_[i] = std::pow(elem_[i], p);
	}
}

template<typename T>
void Vector<T>::log() {
	vr_log(nRows_, elem_ , elem_);
}

template<typename T>
void Vector<T>::abs() {
# pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		elem_[i] = std::abs(elem_[i]);
}

/**	Set all elements to 0.
 *
 *	Set all elements in the vector to 0.
 */
template<typename T>
void Vector<T>::setToZero() {
	memset(elem_, 0, nRows_ * sizeof(T));
}

/**	Set all elements to a constant value.
 *
 *	Set all elements in the vector to a constant value.
 */
template<typename T>
void Vector<T>::fill(const T value) {
	//fill the array with the constant
	std::fill(begin(), end(), value);
}

/**
 *	Set all elements to be >= threshold
 */
template<typename T>
void Vector<T>::ensureMinimalValue(const T threshold) {
	for (u32 row = 0; row < nRows_; row++) {
		if (at(row) < threshold)
			at(row) = threshold;
	}
}

template<typename T>
Vector<T> &Vector<T>::operator=(Vector<T> rhs) {
	swap(rhs);
	return *this;
}

template<typename T>
T Vector<T>::sum() const {
	T result = 0;
	for (u32 i = 0; i < nRows_; i++) {
		result += elem_[i];
	}
	return result;
}

template<typename T>
void Vector<T>::addSummedColumns(const Matrix<T>& matrix, const T scale) {
	require(!matrix.needsU64Space_);
	require_eq(matrix.nRows(), nRows_);
	// TODO parallelize
	for (u32 row = 0; row < matrix.nRows(); row++) {
		for (u32 column = 0; column < matrix.nColumns(); column++)
			at(row) += scale * matrix.at(row, column);
	}
}
template<typename T>
void Vector<T>::addSummedColumnsChannelWise(const Matrix<T>& matrix, const u32 channels, const T scale)
{
	require(!matrix.needsU64Space_);
	require_eq(matrix.nRows() % channels, 0);
	require_eq(channels, nRows_);

	u32 channelSize = matrix.nRows() / channels;
	for(u32 i = 0; i < matrix.nRows(); i++) {
		for(u32 column = 0; column < matrix.nColumns(); column++) {
			at(i / channelSize) += scale * matrix.at(i, column);
		}
	}
}
template<typename T>
void Vector<T>::addSquaredSummedColumns(const Matrix<T>& matrix, const T scale) {
	require(!matrix.needsU64Space_);
	require_eq(matrix.nRows(), nRows_);
	// TODO parallelize
	for (u32 row = 0; row < matrix.nRows(); row++) {
		for (u32 column = 0; column < matrix.nColumns(); column++)
			at(row) += scale * matrix.at(row, column) * matrix.at(row, column);
	}
}

template<typename T>
void Vector<T>::addSummedRows(const Matrix<T>& matrix, const T scale) {
	require(!matrix.needsU64Space_);
	require_eq(matrix.nColumns(), nRows_);
	// TODO parallelize
	for (u32 column = 0; column < matrix.nColumns(); column++) {
		for (u32 row = 0; row < matrix.nRows(); row++) {
			at(column) += scale * matrix.at(row, column);
		}
	}
}

template<typename T>
T Vector<T>::chiSquareDistance(const Vector<T> &v) const {
	require(this->nRows_ == v.nRows_);
	Vector<T> tmp(this->nRows_);
#pragma omp parallel for
	for (u32 i = 0; i < this->nRows_; i++) {
		// handle special case: this->at(i) + v->at(i) == 0
		if (this->at(i) + v.at(i) == 0) {
			tmp.at(i) = 0;
		}
		else {
			tmp.at(i) = 0.5 * (this->at(i) - v.at(i)) * (this->at(i) - v.at(i)) / (this->at(i) + v.at(i));
		}
	}
	return tmp.l1norm();
}

template<typename T>
T Vector<T>::modifiedChiSquareDistance(const Vector<T> &v) const {
	require(this->nRows_ == v.nRows_);
	Vector<T> tmp(this->nRows_);
#pragma omp parallel for
	for (u32 i = 0; i < this->nRows_; i++) {
		// handle special case: this->at(i) + v->at(i) == 0
		if (this->at(i) + v.at(i) == 0) {
			tmp.at(i) = 0;
		}
		else {
			tmp.at(i) = 2 * this->at(i) * v.at(i) / (this->at(i) + v.at(i));
		}
	}
	return tmp.l1norm();
}

template<typename T>
void Vector<T>::chiSquareFeatureMap(const Vector<T>& v, u32 n, T samplingDistance) {
	require_eq(v.nRows() * (2 * n + 1), nRows());
#pragma omp parallel for
	for (u32 i = 0; i < v.nRows(); i++) {
		T x = std::max(v.at(i), Types::absMin<T>());
		u32 baseIndex = i * (2 * n + 1);
		for (u32 j = 0; j < 2 * n + 1; j++) {
			if (j == 0) { // j = 0
				at(baseIndex) = std::sqrt(samplingDistance * x);
			}
			else if (j % 2 == 1) { // j odd
				T kappa = 1.0 / std::cosh(M_PI * (j+1)/2 * samplingDistance);
				at(baseIndex + j) = std::sqrt(2 * kappa * samplingDistance * x) *
						std::cos((j+1)/2 * samplingDistance * std::log(x));
			}
			else { // j even
				T kappa = 1.0 / std::cosh(M_PI * j/2 * samplingDistance);
				at(baseIndex + j) = std::sqrt(2 * kappa * samplingDistance * x) *
						std::sin(j/2 * samplingDistance * std::log(x));
			}
		}
	}
}

template<typename T>
void Vector<T>::getMaxOfColumns(const Matrix<T> &X){
	require(!X.needsU64Space_);
	// TODO parallelize
	require_eq(X.nColumns(), nRows_);
	for (u32 j = 0; j < X.nColumns(); j++)
		at(j) = *std::max_element(&X.at(0,j), &X.at(0,j) + X.nRows());
}

template<typename T>
void Vector<T>::swap(Vector<T> &vector){
	std::swap(nRows_, vector.nRows_);
	std::swap(nAllocatedCells_, vector.nAllocatedCells_);
	std::swap(elem_, vector.elem_);
}

template<typename T>
void Vector<T>::swap(Matrix<T> &X){
	require(!X.needsU64Space_);
	u32 nRows = X.nRows_ * X.nColumns_;
	X.nRows_ = nRows_;
	X.nColumns_ = 1;
	nRows_ = nRows;
	u32 tmpAllocatedCells = (u32)X.nAllocatedCells_;
	X.nAllocatedCells_ = nAllocatedCells_;
	nAllocatedCells_ = tmpAllocatedCells;
	std::swap(elem_, X.elem_);
}

template<typename T>
std::string Vector<T>::toString(bool transpose) const {
	require(nRows_ > 0);
	std::stringstream s;
	if (transpose) {
		for (u32 j = 0; j < nRows_ - 1; j++)
			s << at(j) << " ";
		s << at(nRows_ - 1);
	}
	else {
		for (u32 j = 0; j < nRows_ - 1; j++)
			s << at(j) << std::endl;
		s << at(nRows_ - 1);
	}
	return s.str();
}

template<typename T>
void Vector<T>::writeBinaryHeader(Core::IOStream& stream) {
	require(stream.is_open());
	stream << nRows_;
}

template<typename T>
void Vector<T>::writeAsciiHeader(Core::IOStream& stream) {
	require(stream.is_open());
	stream << nRows_ << Core::IOStream::endl;
}

template<typename T>
void Vector<T>::writeBinary(Core::IOStream& stream) {
	require(stream.is_open());
	for (u32 row = 0; row < nRows_; row++) {
		stream << (f32)at(row);
	}
}

template<typename T>
void Vector<T>::writeAscii(Core::IOStream& stream, bool scientific) {
	require(stream.is_open());
	if (scientific)
		stream << Core::IOStream::scientific;
	for (u32 row = 0; row < this->nRows_ - 1; row++) {
		stream << at(row) << " ";
	}
	stream << at(this->nRows_ - 1) << Core::IOStream::endl;
}

template<typename T>
void Vector<T>::write(const std::string& filename, bool scientific) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::out);
		writeBinaryHeader(stream);
		writeBinary(stream);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::out);
		writeAsciiHeader(stream);
		writeAscii(stream, scientific);
		stream.close();
	}
	else {
		Core::AsciiStream stream(filename, std::ios::out);
		writeAsciiHeader(stream);
		writeAscii(stream, scientific);
		stream.close();
	}
}

template<typename T>
void Vector<T>::readHeader(Core::IOStream& stream) {
	require(stream.is_open());
	u32 nRows;
	stream >> nRows;
	resize(nRows);
}

template<typename T>
void Vector<T>::read(Core::IOStream& stream) {
	f32 x;
	for (u32 row = 0; row < this->nRows_; row++) {
		stream >> x;
		at(row) = (T)x;
	}
}

template<typename T>
void Vector<T>::read(const std::string& filename) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::in);
		readHeader(stream);
		read(stream);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::in);
		readHeader(stream);
		read(stream);
		stream.close();;
	}
	else {
		Core::AsciiStream stream(filename, std::ios::in);
		readHeader(stream);
		read(stream);
		stream.close();
	}
}

} // namespace Math

#endif /* MATH_VECTOR_HH_ */
