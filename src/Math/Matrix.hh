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

#ifndef MATH_MATRIX_HH_
#define MATH_MATRIX_HH_

#include <Core/CommonHeaders.hh>
#include <Core/OpenMPWrapper.hh>

#include <Math/Blas.hh>
#include <Math/Vector.hh>   		// for matrix-vector operations (Blas 2)
#include <Math/FastVectorOperations.hh>
#include <Math/Random.hh>

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>				// to use std::cout
#include <typeinfo>
#include <sstream>

namespace Math {

template<typename T>
class Vector;

/*
 * matrix with col-major storage
 * use of BLAS routines
 *
 */

template<typename T>
class Matrix {
	friend class Vector<T>;
	friend class Vector<f64>;
	friend class Vector<f32>;
	friend class Matrix<f64>;
	friend class Matrix<f32>;
	friend class Vector<u32>;
protected:
	// the number of sizeof(T) elements that are actually allocated
	// (may differ from nRows_ * nColumns_ due to lazy resize)
	u64 nAllocatedCells_;
	u32 nRows_;
	u32 nColumns_;
	bool needsU64Space_;
	T *elem_;
protected:
	static bool initialized;
	static s32 maxThreads;
	static s32 _initialize();
	static s32 initialize();
	s32 nThreads_;
public:
	s32 getNumberOfThreads(){ return nThreads_; }
public:
	// iterators
	typedef T* iterator;
	typedef const T* const_iterator;
	iterator begin() { return elem_; }
	const_iterator begin() const { return elem_; }
	iterator end() { return &(elem_[(u64)nRows_ * (u64)nColumns_]); }
	const_iterator end() const { return &(elem_[(u64)nRows_ * (u64)nColumns_]); }
public:
	// constructor with memory allocation
	Matrix(u32 nRows = 0, u32 nColumns = 0);

	// (deep) copy constructor
	Matrix(const Matrix<T> &X);

	// constructor for creating sub-matrices via copyBlockFromMatrix()
	Matrix(const Matrix<T> &X, u32 rowIndexX, u32 colIndexX,
			u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns);

	// destructor
	virtual ~Matrix();

	T& operator() (s32 row, s32 column) { return at(row, column); }

private:
	bool allocate();

private:
	void writeBinaryHeader(Core::IOStream& stream, bool transpose);
	void writeAsciiHeader(Core::IOStream& stream, bool transpose);
	void writeBinary(Core::IOStream& stream, bool transpose);
	void writeAscii(Core::IOStream& stream, bool transpose, bool scientific);
	void readHeader(Core::IOStream& stream, bool transpose);
	void read(Core::IOStream& stream, bool transpose = false);

public:
	// file IO
	void write(const std::string& filename, bool transpose = false, bool scientific = false);
	void read(const std::string& filename, bool transpose = false);

public:
	// free memory
	void clear();

	// required for assignment operator
	void swap(Matrix<T> &X);

	// swap matrix and vector (matrix will end up being a single column, former matrix columns are concatenated in the vector)
	void swap(Vector<T> &X);

	void setNumberOfThreads(int nThreads) { nThreads_ = nThreads;};

	// need assignment operator, because we have a copy constructor
	// pass by value ! (needed for temporary object creation)
	Matrix<T> & operator=(Matrix<T> X);

	// resize & allocate
	// side effect: after resize content is meaningless
	// if reallocate is true enforce reallocation of memory
	virtual void resize(u32 nRows, u32 nColumns, bool reallocate = false);

	// resize at most to a size that has been used before
	// -> no new memory allocation, old content remains valid
	void safeResize(u32 nRows, u32 nColumns);

	virtual void reshape(u32 nRows, u32 nColumns);

	void setVisibleColumns(u32 nColumns) { safeResize(nRows_, nColumns); }

	// set dimensions to those of X and allocate
	template <typename S>
	void copyStructure(const Matrix<S> &X);

	// returns the number of rows
	u32 nRows() const { return nRows_; }

	// copy method
	// this = X
	// for matrices with same dimensions
	template<typename S>
	void copy(const Matrix<S> &X);

	// copy method
	// this = X
	// array X is assumed to be of size nRows_ * nColumns_
	template<typename S>
	void copy(const S *X, u32 rowOffset = 0, u32 colOffset = 0);

	// copy from std::vector
	template<typename S>
	void copy(const std::vector<S> &X, u32 rowOffset = 0, u32 colOffset = 0);

	// returns the number of columns
	u32 nColumns() const { return nColumns_; }

	// returns whether the matrix is empty
	bool empty() const { return nRows() == 0 || nColumns() == 0; }

	// returns whether all matrix entries are finite
	bool isFinite() const;

	// returns the total number of entries
	u32 size() const { require(!needsU64Space_); return nRows_*nColumns_; }

	bool needsU64Space() { return needsU64Space_; }

	// fills the matrix with the given value
	void fill(T value) { std::fill(elem_, elem_ + (u64)nRows_ * (u64)nColumns_, value); }


	// fills the matrix from position (rowA, columnA) to (rowB, columnB) with the value
	void fill(u32 rowA, u32 columnA, u32 rowB, u32 columnB, T value);

	//performs convolution
	void prepareConvolution(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight,
			const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY);
	void prepareConvolutionBackProp(const Matrix<T>& source, const u32 destWidth, const u32 destHeight,
			const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight);
	void prepareConvolutionSame(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight,
			const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY);
	void prepareConvolutionSameBackProp(const Matrix<T>& source, const u32 destWidth, const u32 destHeight,
				const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY);
	void rearrange(const Matrix<T>& source, const u32 numImages);
	void rearrangeBackProp(const Matrix<T>& source, const u32 channels);
	//performs max pooling both forward and backward
	void maxPool(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight, const u32 sourceChannels,
			const u32 poolSize, const u32 stride);
	void backPropogateMaxPool(const Matrix<T>& activationIn, const Matrix<T>& activationOut,
			const Matrix<T>& errorSignalOut, const u32 sourceWidth, const u32 sourceHeight,
			const u32 sourceChannels, const u32 poolSize, const u32 stride);
	//performs average pooling
	void avgPool(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight, const u32 sourceChannels,
			const u32 poolSize, const u32 stride);
	void backPropogateAvgPool(const Matrix<T>& errorSignalOut, const u32 sourceWidth, const u32 sourceHeight,
				const u32 sourceChannels, const u32 poolSize, const u32 stride);
	/////////////////////////////////

	// set all values < threshold to threshold
	void ensureMinimalValue(const T threshold);

	// set all values > threshold to threshold
	void ensureMaximalValue(const T threshold);

	// get reference to element in row i, column j
	T& at(u32 i, u32 j){
		require(i < nRows_);
		require(j < nColumns_);
		return *(elem_ + (u64)j*(u64)nRows_ + i);
	}

	// get const reference to element in row i, column j
	const T& at(u32 i, u32 j) const {
		require(i < nRows_);
		require(j < nColumns_);
		return *(elem_ + (u64)j*(u64)nRows_ + i);
	}

	// get value of the element in row i, column j
	const T get(u32 i, u32 j) const {
		require(i < nRows_);
		require(j < nColumns_);
		return at(i, j);
	}

	// convert matrix to string
	std::string toString(bool transpose = false) const;

	// get row with index rowIndex
	void getRow(u32 rowIndex, Math::Vector<T> &row) const;

	// get column with index columnIndex
	void getColumn(u32 columnIndex, Math::Vector<T> &column) const;

	// set row at rowIndex to values in vector row
	void setRow(u32 rowIndex, const Math::Vector<T> &row);

	// set column at columnIndex to values in vector column
	void setColumn(u32 columnIndex, const Math::Vector<T> &column);

	// copy block from matrix to specific position
	void copyBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns = 1);

	// add block from matrix to specific position
	void addBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns, T scale = 1.0);

	// this = 0
	void setToZero() { memset(elem_, 0, (u64)nRows_ * (u64)nColumns_ * sizeof(T)); }



public:

	/*
	 * MATH OPERATIONS
	 *
	 */

	/*
	 * Blas1-like methods
	 */

	// this += alpha * X
	template<typename S>
	void add(const Matrix<S> &X, S alpha = 1.0);

	// this += weights * X, weights each column with a weight
	void addWeighted(const Matrix<T>& X, const Vector<T>& weights);

	// @return l1-norm of matrix
	T l1norm() const { require(!needsU64Space_); return Math::mt_asum(nRows_ * nColumns_, elem_, nThreads_);}

	// @return sum of squared matrix entries
	T sumOfSquares() const { require(!needsU64Space_); return dot (*this); }

	// dot product
	// return this' * X
	// for matrices with multiple columns: interpret matrix as vector
	T dot(const Matrix<T> &X) const;

	// scale elements
	// this *= alpha
	void scale(T alpha);

	/*
	 * Blas2-like methods
	 */

	// rank-1 update: this += alpha * x y^T
	void addOuterProduct(const Vector<T>& x, const Vector<T> &y, T alpha, u32 lda = 0);

	/*
	 * Blas3-like methods
	 */

	// (*this) = (scaleA * matrixA) * matrixB + scaleC * (*this)
	void addMatrixProduct(const Matrix<T> &matrixA, const Matrix<T> &matrixB,
			T scaleC = 0, T scaleA = 1, bool transposeA = false, bool transposeB = false);

	/*
	 * special methods required for neural network computations
	 */

	// apply sigmoid function to each element of matrix
	// this = 1.0 / (1.0 + exp(-gamma * this))
	void sigmoid(T gamma = 1.0);

	// apply triangle function f(x) = |x| if -1 <= x <= 1, else 0 to each element of matrix
	void triangle();

	// apply softmax to each column of matrix
	void softmax();

	// return sum over all elements
	T sum() const;

	// apply max to each column of matrix (set maximal element in column to 1, all other elements to 0)
	void max();

	// set result to elementwise max(A,B)
	void max(const Matrix<T> &A, const Matrix<T> &B);

	// elementwise multiplication with Kronecker delta: this->at(i,j) *= (A.at(i,j) == B.at(i,j)) ? 1.0 : 0.0
	void elementwiseMultiplicationWithKroneckerDelta(const Matrix<T> &A, const Matrix<T> &B);

	// copy nClones of X to *this, the copies are appended in vertical direction
	void clone(const Matrix<T> &X, u32 nClones);

	// replace each matrix element by nClones clones of itself, clones are added vertically (nColumns stays constant)
	void cloneElementwise(const Matrix<T> &X, u32 nClones);

	// for each column c: this->at(i,c) += X.at(j,c) for all j with j % this->nRows_ == i
	void addElementsByModuloIndex(const Matrix<T> &X);

	// compute a finite feature map from X to approximate the feature map of the chi-square kernel (applied to each column)
	// cf. Efficient Additive Kernels via Explicit Feature Maps, Vedaldi and Zisserman
	// approximation based of discrete fourier transform with 2n+1 sampling points
	void chiSquareFeatureMap(const Matrix<T> &X, u32 n, T samplingDistance);

	// compute a finite feature map from X to approximate the feature map of the histogram intersection kernel (applied to each column)
	// cf. Efficient Additive Kernels via Explicit Feature Maps, Vedaldi and Zisserman
	// approximation based of discrete fourier transform with 2n+1 sampling points
	void histogramIntersectionFeatureMap(const Matrix<T> &X, u32 n, T samplingDistance);

	void elementwiseMultiplicationWithApproximateFeatureMapDerivative(const Matrix<T> &X, u32 n, T samplingDistance, T kappa0 = 1.0);

	// this = this .* (X .* (1 - X))
	void elementwiseMultiplicationWithSigmoidDerivative(const Matrix<T> &X);

	// this = this .* (|X| <= 1 ? sign(X) : 0)
	void elementwiseMultiplicationWithTriangleDerivative(const Matrix<T> &X);

	// this = this .* (1 - X .^ 2)
	void elementwiseMultiplicationWithTanhDerivative(const Matrix<T> &X);

	// for each column i: this(_,i) = (diag(softmax(_,i)) - softmax(_,i)*softmax(_,i)^T) * this(_,i)
	void multiplicationWithSoftmaxDerivative(const Matrix<T> &softmax);

	// this = this .* (X < thresoldLeft || X > thresholdRight ? 0 : 1)
	void elementwiseMultiplicationWithClippedDerivative(const Matrix<T> &X, T thresholdLeft, T thresholdRight);

	// this = this ./ exp(x)
	void elementwiseMultiplicationWithLogDerivative(const Matrix<T> &X);

	// this = this .* p * |x|^p
	void elementwiseMultiplicationWithSignedPowDerivative(const Matrix<T> &X, T p);

	// for each column i: this(_,i) = 1/norm(i) * (I - l2Norm(_,i)*l2Norm(_,i)^T) * this(_,i)
	void multiplicationWithL2NormalizationDerivative(const Matrix<T> &X, const Vector<T> &norm);

	// sum nNeighbors of one row to a single value and add it to corresponding element of *this
	// requires this->nRows() * nNeighbors= X.nRows
	void addSummedNeighborsInARow(const Matrix<T> &X, u32 nNeighbors);

	// return number of classifications errors; each column of *this is interpreted as a probability distribution
	u32 nClassificationErrors(const Matrix<T>& targets) const;

	// return the value of the cross entropy objective function; each column of *this is interpreted as a probability distribution
	T crossEntropyObjectiveFunction(const Matrix<T>& targets) const;

	// return the value of the weighted cross entropy objective function; each column of *this is interpreted as a probability distribution
	T weightedCrossEntropyObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const;

	// return the value of the squared error objective function
	T squaredErrorObjectiveFunction(const Matrix<T>& targets) const;

	// return the value of the weighted squared error objective function
	T weightedSquaredErrorObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const;

	// return the value of the smoothed l1 objective function
	T smoothedL1ObjectiveFunction(const Matrix<T>& targets) const;

	// return the value of the weighted smoothed l1 objective function
	T weightedSmoothedL1ObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const;

	// dot product
	T dotWithColumn(const Vector<T> &v, u32 thisColumnIndex) const;

	// expand by second order polynomial features (column-wise)
	void setToSecondOrderFeatures(const Matrix<T> &X);

	// expand by diagonal second order polynomial features (column-wise) x_1,...x_n,x_1^2,...,x_n^2
	void setToDiagonalSecondOrderFeatures(const Matrix<T> &X);

	// expand by second and third order polynomial features (column-wise)
	void setToThirdOrderFeatures(const Matrix<T> &X);

	// expand by diagonal second and third order polynomial features (column-wise)
	void setToDiagonalThirdOrderFeatures(const Matrix<T> &X);

	// compute Gaussian mixture posteriors
	void gaussianMixturePosteriors(const Matrix<T> &X, const Matrix<T> &means, const Matrix<T> &variances, const Vector<T> &weights);

	// apply fisher vector encoding to each column of the matrix
	void fisherEncoding(const Matrix<T> &X, const Matrix<T> &means, const Matrix<T> &variances, const Vector<T> &weights);

	// apply dropout to matrix, each element is dropped with probability dropoutProbability
	void dropout(const T dropoutProbability);

	void rpropUpdate(const Matrix<T> &newGradients, Matrix<T> &oldGradients, Matrix<T> &updateValues, const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue);


	/*
	 * more math operations
	 */

	// apply tanh to each element of matrix
	void tanh();

	// apply exp to each element of matrix
	void exp();

	// apply power function to the absolute value of each element of matrix, keep original sign of each value
	void signedPow(T p);

	// apply log to each element of matrix
	void log();

	// apply sin to each element of matrix
	void sin();

	// apply cos to each element of matrix
	void cos();

	// apply arc sin to each element of matrix
	void asin();

	// apply arc cos to each element of matrix
	void acos();

	// absolute values
	void abs();

	// maximal value in matrix
	T maxValue() const;

	// return index of minimum absolute value in column
	u32 argAbsMin(u32 column) const;

	// return index of maximum absolute value in column
	u32 argAbsMax(u32 column) const;

	// save arg max of each column of *this in the rows of v
	template<typename S>
	void argMax(Vector<S>& v) const;

	// this = this .* X
	void elementwiseMultiplication(const Matrix<T> &X);

	// this = this ./ X
	void elementwiseDivision(const Matrix<T> &X);

	// add constant value c to each element
	void addConstantElementwise(T c);

	// add vector (scaled by alpha) to column with given index
	void addToColumn(const Vector<T> &v, u32 column, T alpha = 1.0);

	// multiply column by scalar alpha
	void multiplyColumnByScalar(u32 column, T alpha);

	// multiply row by scalar alpha
	void multiplyRowByScalar(u32 row, T alpha);

	// add vector (scaled by alpha) to row with given index
	void addToRow(const Vector<T> &v, u32 row, T alpha = 1.0);

	// add vector (scaled by alpha) to each column of the matrix
	void addToAllColumns(const Vector<T> &v, T alpha = 1.0);

	// add vector (scaled by alpha) to each channel (one element of vector to one channel)
	void addToAllChannels(const Vector<T> &v, const u32 channels, T alpha = 1.0);

	// add vector (scaled by alpha) to each column of the matrix
	void addToAllRows(const Vector<T> &v, T alpha = 1.0);

	// for each i: multiply column i by scalars[i]
	void multiplyColumnsByScalars(const Vector<T> &scalars);

	// for each i: divide column i by scalars[i]
	void divideColumnsByScalars(const Vector<T> &scalars);

	// for each i: multiply row i by scalars[i]
	void multiplyRowsByScalars(const Vector<T> &scalars);

	// for each i: multiply row i by scalars[i]
	void divideRowsByScalars(const Vector<T> &scalars);

private:
	// this = \alpha A^{"", T} * B^{"", T} + \gamma this
	static void _gemm(bool transposeA, bool transposeB, u32 M, u32 N, u32 K, T scaleA, T* matrixA, u32 lda, T* matrixB, u32 ldb, T scaleC, T* matrixC, u32 ldc);

	void appendSecondOrderFeatures(const Matrix<T> &X, u32 offset);

	void appendDiagonalSecondOrderFeatures(const Matrix<T> &X, u32 offset);

	void appendThirdOrderFeatures(const Matrix<T> &X, u32 offset);

	void appendDiagonalThirdOrderFeatures(const Matrix<T> &X, u32 offset);
};


template<typename T>
bool Matrix<T>::initialized = false;

template<typename T>
s32 Matrix<T>::maxThreads = 1;

template<typename T>
s32 Matrix<T>::_initialize(){
	if (!initialized){
		initialized = true;

		int value;

		char* svalue;
		svalue = std::getenv("OMP_NUM_THREADS");
		if (svalue != NULL){
			std::string tmp = svalue;
			std::istringstream ss(tmp);
			ss >> value;
			if (ss.fail())
				value = 1;
		}
		else{
			value = 1;
		}

		maxThreads = value;
		Core::omp::set_num_threads(value);
		std::cout << "Maximum number of threads for CPU matrix operations: " << maxThreads << std::endl;

	}
	return maxThreads;
}

template<typename T>
s32 Matrix<T>::initialize(){
	if (!initialized){
		// ensure that initialize is in fact only invoked once
		maxThreads = Matrix<u32>::_initialize();
		initialized = true;
	}
	return maxThreads;
}

/**	Allocate the memory for the matrix.
 *
 * 	Allocate the memory for the matrix. If the size is 0 the pointer
 * 	is ZERO.
 */
template<typename T>
bool Matrix<T>::allocate() {
	if (elem_)
		delete [] elem_;
	elem_ = (u64)nRows_ * (u64)nColumns_ > 0 ? new T[(u64)nRows_*(u64)nColumns_] : 0;
	nAllocatedCells_ = (u64)nRows_ * (u64)nColumns_ > 0 ? (u64)nRows_*(u64)nColumns_ : 0;
	return true;
}

// constructor with allocation
template<typename T>
Matrix<T>::Matrix(u32 nRows, u32 nColumns) :
nAllocatedCells_(0),
nRows_(nRows),
nColumns_(nColumns),
needsU64Space_((u64)nRows * (u64)nColumns > (u64)Types::max<u32>()),
elem_(0),
nThreads_(1)
{
	nThreads_ = initialized ? maxThreads : initialize();
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
}

// copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &X) :
nAllocatedCells_(0),
nRows_(X.nRows_),
nColumns_(X.nColumns_),
needsU64Space_(X.needsU64Space_),
elem_(0),
nThreads_(1)
{
	nThreads_ = initialized ? maxThreads : initialize();
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
	if (!needsU64Space_) {
		copy(X);
	}
	else {
#pragma omp parallel for
		for (u32 row = 0; row < nRows_; row++) {
			for (u32 col = 0; col < nColumns_; col++) {
				at(row, col) = X.at(row, col);
			}
		}
	}
}

// copy constructor for sub-matrices
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &X, u32 rowIndexX, u32 colIndexX,
		u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns) :
		nAllocatedCells_(0),
		nRows_(nRows),
		nColumns_(nColumns),
		needsU64Space_((u64)nRows * (u64)nColumns > (u64)Types::max<u32>()),
		elem_(0),
		nThreads_(initialized ? maxThreads : initialize())
{
	if ((u64)nRows_* (u64)nColumns_ < 250000)
		nThreads_ = 1;
	allocate();
	copyBlockFromMatrix(X, rowIndexX, colIndexX, thisRowIndex, thisColIndex, nRows, nColumns);
}

template<typename T>
Matrix<T>::~Matrix() {
	if (elem_)
		delete [] elem_;
	elem_ = 0;
}

template<typename T>
void Matrix<T>::clear() {
	if (elem_)
		delete [] elem_;
	elem_ = 0;
	nRows_ = 0;
	nColumns_ = 0;
	nAllocatedCells_ = 0;
	needsU64Space_ = false;
}


template<typename T>
void Matrix<T>::swap(Matrix<T> &X){
	std::swap(nRows_, X.nRows_);
	std::swap(nColumns_, X.nColumns_);
	std::swap(nAllocatedCells_, X.nAllocatedCells_);
	std::swap(needsU64Space_, X.needsU64Space_);
	std::swap(elem_, X.elem_);
}

template<typename T>
void Matrix<T>::swap(Vector<T> &X){
	require(!needsU64Space_);
	u32 nRows = X.nRows_;
	X.nRows_ = nRows_ * nColumns_;
	nRows_ = nRows;
	nColumns_ = 1;
	u32 tmpAllocatedCells = X.nAllocatedCells_;
	X.nAllocatedCells_ = (u32)nAllocatedCells_;
	nAllocatedCells_ = (u64)tmpAllocatedCells;
	std::swap(elem_, X.elem_);
}

template<typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> rhs) {
	require(!needsU64Space_);
	swap(rhs);
	return *this;
}

template<typename T>
void Matrix<T>::resize(u32 nRows, u32 nColumns, bool reallocate) {
	reallocate |= (u64)nRows * (u64)nColumns > nAllocatedCells_;
	nRows_ = nRows;
	nColumns_ = nColumns;
	if (reallocate) {
		needsU64Space_ = (u64)nRows * (u64)nColumns > (u64)Types::max<u32>();
		allocate();
	}
}

template<typename T>
void Matrix<T>::safeResize(u32 nRows, u32 nColumns) {
	require_le((u64)nRows * (u64)nColumns, nAllocatedCells_);
	resize(nRows, nColumns, false);
}

template<typename T>
void Matrix<T>::reshape(u32 nRows, u32 nColumns) {
	require_eq(nRows_ * nColumns_, nRows * nColumns);
	nRows_ = nRows;
	nColumns_ = nColumns;
}

template<typename T> template <typename S>
void Matrix<T>::copyStructure(const Matrix<S> &X) {
	resize(X.nRows(), X.nColumns());
}

template<typename T>
bool Matrix<T>::isFinite() const {
	require(!needsU64Space_);
	for (u32 row = 0; row < nRows_; row++){
		for (u32 column = 0; column < nColumns_; column++){
			T val = at(row, column);
			if (Types::isNan(val) || val > Types::max<T>() || val < Types::min<T>())
				return false;
		}
	}
	return true;
}
template<typename T>
void Matrix<T>::avgPool(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight, const u32 sourceChannels,
			const u32 poolSize, const u32 stride) {
	require_eq(source.nRows_,
			sourceChannels * sourceHeight * sourceWidth);
	require_gt(stride, 0);
	require_gt(poolSize, 0);
	require_ge(sourceWidth, poolSize);
	require_ge(sourceHeight, poolSize);
	require_ge(sourceWidth, stride);
	require_ge(sourceHeight, stride);
	require_eq(nColumns_, source.nColumns_);

	u32 resultWidth = (u32)ceil((f64)sourceWidth / (f64)stride);
	u32 resultHeight = (u32)ceil((f64)sourceHeight / (f64)stride);
	require_eq(nRows_, resultWidth * resultHeight * sourceChannels);
	u32 pixelX, pixelY;
	T sum;
	s32 index = -1;
	for(u32 img = 0; img < nColumns_; img++) {
		for(u32 ch=0; ch < sourceChannels; ch++) {
			for(u32 rPixelX = 0; rPixelX < resultWidth; rPixelX++) {
				for(u32 rPixelY = 0; rPixelY < resultHeight; rPixelY++) {
					sum = 0;
					pixelX = rPixelX*stride;
					pixelY = rPixelY*stride;
					T num = 0;
					for(u32 m=pixelX; (m<pixelX+poolSize) && (m<sourceWidth); m++) {
						for(u32 n=pixelY; (n<pixelY+poolSize) && (n<sourceHeight); n++) {
							index = ch*sourceWidth*sourceHeight +
									m * sourceHeight + n;
							sum += source.at(index, img);
							num += 1;
						}
					}
					this->at(ch*resultHeight*resultWidth + rPixelX * resultHeight + rPixelY,img) = sum / (poolSize * poolSize);
				}
			}
		}
	}
}
template<typename T>
void Matrix<T>::backPropogateAvgPool(const Matrix<T>& errorSignalOut, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 poolSize, const u32 stride) {

	require_eq(nRows_, sourceWidth * sourceHeight * sourceChannels);
	require_gt(poolSize, 0);
	require_ge(sourceWidth, poolSize);
	require_ge(sourceHeight, poolSize);
	require_gt(stride, 0);
	require_ge(sourceWidth, stride);
	require_ge(sourceHeight, stride);
	u32 errorSignalWidth = ceil((f64)sourceWidth/(f64)stride);
	u32 errorSignalHeight = ceil((f64)sourceHeight/(f64)stride);
	require_eq(nColumns_, errorSignalOut.nColumns_);

	require_eq(errorSignalOut.nRows_, errorSignalHeight * errorSignalWidth * sourceChannels);

	s32 indexInErrorSignal = -1;


	//loops through all images in a batch
	for(u32 img=0; img<nColumns_; img++) {
		for(u32 ch=0; ch<sourceChannels; ch++) {
			for(u32 pixelX=0; pixelX<sourceWidth; pixelX++) {
				for(u32 pixelY=0; pixelY<sourceHeight; pixelY++) {

					//calculates start of the first grid containing current Pixel
					u32 gridStartX = ((s32)pixelX + 1 - (s32)poolSize) < 0 ? 0 :
							(u32)(ceil((f64)((s32)pixelX + 1 - (s32)poolSize)/(f64)stride) * stride);
					u32 gridStartY = ((s32)pixelY + 1 - (s32)poolSize) < 0 ? 0 :
							(u32)(ceil((f64)((s32)pixelY + 1 - (s32)poolSize)/(f64)stride) * stride);
					//////////////////////////////////

					//iterates over all grids containing this pixel
					for(u32 gridX=gridStartX; gridX<=pixelX; gridX+=stride) {
						for(u32 gridY=gridStartY; gridY<=pixelY; gridY+=stride) {

							indexInErrorSignal = ch * errorSignalHeight * errorSignalWidth +
									(gridX/stride) * errorSignalHeight + (gridY/stride);

							this->at(ch*sourceHeight*sourceWidth + pixelX * sourceHeight + pixelY, img)
									= errorSignalOut.at(indexInErrorSignal, img) / (f64)(poolSize * poolSize);
						}
					}
				}
			}
		}
	}
}
template<typename T>
void Matrix<T>::maxPool(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 poolSize, const u32 stride)
{
	require_eq(source.nRows_,
			sourceChannels * sourceHeight * sourceWidth);
	require_gt(stride, 0);
	require_gt(poolSize, 0);
	require_ge(sourceWidth, poolSize);
	require_ge(sourceHeight, poolSize);
	require_ge(sourceWidth, stride);
	require_ge(sourceHeight, stride);
	require_eq(nColumns_, source.nColumns_);
	u32 resultWidth = (u32)ceil((f64)sourceWidth / (f64)stride);
	u32 resultHeight = (u32)ceil((f64)sourceHeight / (f64)stride);
	require_eq(nRows_, resultWidth * resultHeight * sourceChannels);
	u32 pixelX, pixelY;
	T maxValue;
	s32 index = -1;
	for(u32 img=0; img<nColumns_; img++) {
		for(u32 ch=0; ch<sourceChannels; ch++) 	{
			for(u32 rPixelX=0; rPixelX<resultWidth; rPixelX++) {
				for(u32 rPixelY=0; rPixelY<resultHeight; rPixelY++) {
					maxValue = std::numeric_limits<T>::min();
					index = -1;
					pixelX = rPixelX*stride;
					pixelY = rPixelY*stride;
					for(u32 m=pixelX; (m<pixelX+poolSize) && (m<sourceWidth); m++) {
						for(u32 n=pixelY; (n<pixelY+poolSize) && (n<sourceHeight); n++) {
							index = ch*sourceWidth*sourceHeight +
									m * sourceHeight + n;
							if(source.at(index, img) > maxValue) {
								maxValue = source.at(index, img);
							}
						}
					}
					this->at(ch*resultHeight*resultWidth + rPixelX * resultHeight + rPixelY,img) = maxValue;
				}
			}
		}
	}
}
template<typename T>
void Matrix<T>::backPropogateMaxPool(const Matrix<T>& activationIn, const Matrix<T>& activationOut,
		const Matrix<T>& errorSignalOut, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 poolSize, const u32 stride) {
	require_eq(nRows_, sourceWidth * sourceHeight * sourceChannels);
	require_eq(nRows_, activationIn.nRows_);
	require_gt(poolSize, 0);
	require_ge(sourceWidth, poolSize);
	require_ge(sourceHeight, poolSize);
	require_gt(stride, 0);
	require_ge(sourceWidth, stride);
	require_ge(sourceHeight, stride);
	u32 errorSignalWidth = ceil((f64)sourceWidth/(f64)stride);
	u32 errorSignalHeight = ceil((f64)sourceHeight/(f64)stride);
	require_eq(nColumns_, errorSignalOut.nColumns_);
	require_eq(nColumns_, activationIn.nColumns_);
	require_eq(nColumns_, activationOut.nColumns_);
	require_eq(errorSignalOut.nRows_, errorSignalHeight * errorSignalWidth * sourceChannels);
	require_eq(activationOut.nRows_, errorSignalOut.nRows_);
	s32 indexInActivationIn = -1;
	s32 indexInErrorSignal = -1;
	s32 numMaxima = 0;

	//loops through all images in a batch
	for(u32 img=0; img<nColumns_; img++) {
		for(u32 ch=0; ch<sourceChannels; ch++) {
			for(u32 pixelX=0; pixelX<sourceWidth; pixelX++) {
				for(u32 pixelY=0; pixelY<sourceHeight; pixelY++) {
					//calculates start of the first grid containing current Pixel
					u32 gridStartX = ((s32)pixelX + 1 - (s32)poolSize) < 0 ? 0 :
							(u32)(ceil((f64)((s32)pixelX + 1 - (s32)poolSize)/(f64)stride) * stride);
					u32 gridStartY = ((s32)pixelY + 1 - (s32)poolSize) < 0 ? 0 :
							(u32)(ceil((f64)((s32)pixelY + 1 - (s32)poolSize)/(f64)stride) * stride);
					//////////////////////////////////
					indexInActivationIn = ch * sourceHeight * sourceWidth
							+ pixelX * sourceHeight + pixelY;
					for(u32 gridX=gridStartX; gridX<=pixelX; gridX+=stride) {
						for(u32 gridY=gridStartY; gridY<=pixelY; gridY+=stride) {
							indexInErrorSignal = ch * errorSignalHeight * errorSignalWidth +
									(gridX/stride) * errorSignalHeight + (gridY/stride);
							//current pixel is not maximum in current window
							if(activationIn.at(indexInActivationIn,img) != activationOut.at(indexInErrorSignal,img))
								break;
							numMaxima = 0;
							for(u32 i=gridX; (i<(gridX + poolSize)) && i<sourceWidth; i++) {
								for(u32 j=gridY;(j<(gridY+poolSize)) && j<sourceHeight; j++) {
									indexInActivationIn = ch * sourceHeight * sourceWidth +
											i * sourceHeight + j;
									if(activationIn.at(indexInActivationIn, img) == activationOut.at(indexInErrorSignal, img)) {
										numMaxima += 1;
									}
								}
							}
							this->at(ch*sourceHeight*sourceWidth + pixelX * sourceHeight + pixelY, img) +=
									(errorSignalOut.at(indexInErrorSignal, img) / (T) numMaxima);
						}
					}
				}
			}
		}
	}
}
/*
 * Given an image of c,w,h
 * and kernel of size k,k
 * function prepares
 */
//suppose image is of c,w,h and kernel is k,k
//forward version of this function
//will prepare such image for convolution such that
//each result feature is of (w-k+1),(h-k+1)
//so result of forward pass will be of size (w-k+1)*(h-k+1)*k*k*c
template<typename T>
void Matrix<T>::prepareConvolution(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight,
			const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY)
{
	require_eq(nColumns_, source.nColumns_);
	require_eq(source.nRows_, sourceChannels * sourceWidth * sourceHeight);
	require_eq(kernelHeight % 2, 1);
	require_eq(kernelWidth % 2, 1);
	s32 destWidth = (s32)ceil((f32)((s32)sourceWidth - (s32)kernelWidth + 1) / (f32)strideX);
	s32 destHeight = (s32)ceil((f32)((s32)sourceHeight - (s32)kernelHeight + 1) / (f32)strideY);
	s32 sizeDestCh =  destWidth * destHeight;
	require_eq(nRows_, sourceChannels * sizeDestCh * kernelWidth * kernelHeight);
	s32 pixelNum, channelNum, neighbNum, pixelX, pixelY, neighbX, neighbY;
	s32 kernelMiddleX = kernelWidth / 2;
	s32 kernelMiddleY = kernelHeight / 2;

	for(u32 img=0; img<nColumns_; img++) {
		for(u32 j=0; j<nRows_; j++) {
			pixelNum = j / (kernelHeight * kernelWidth * sourceChannels);
			pixelX = (pixelNum / destHeight) * strideX + kernelMiddleX;
			pixelY = (pixelNum % destHeight) * strideY + kernelMiddleY;
			channelNum = j % (kernelHeight * kernelWidth * sourceChannels);
			neighbNum = channelNum % (kernelHeight * kernelWidth);
			channelNum = channelNum / (kernelWidth * kernelHeight);
			neighbX = (neighbNum / kernelHeight) - kernelMiddleX;
			neighbY = (neighbNum % kernelHeight) - kernelMiddleY;
			this->at(j,img) = source.at(channelNum * sourceWidth * sourceHeight +
					(pixelX + neighbX) * sourceHeight + (pixelY + neighbY),img);
		}
	}
}
template<typename T>
void Matrix<T>::prepareConvolutionBackProp(const Matrix<T>& source, const u32 destWidth, const u32 destHeight,
		const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight)
{
	require_eq(nColumns_, source.nColumns_);
	require_eq(nRows_, destWidth * destHeight * destChannels);
	require_eq(kernelWidth % 2, 1);
	require_eq(kernelHeight % 2, 1);
	require_eq(source.nRows_,
			(destWidth - kernelWidth + 1) * (destHeight - kernelHeight + 1) *
			destChannels * kernelWidth * kernelHeight);
	s32 sourceHeight = (destHeight - (s32)kernelHeight + 1);
	s32 pixelX, pixelY, ch, pixelNum, gridStartX, gridStartY, neighNum;
	for(u32 img = 0; img<nColumns_; img++) {
		for(u32 i=0; i<nRows_; i++) {
			this->at(i, img) = 0;
			ch = i / (destHeight * destWidth);
			pixelNum = i % (destHeight * destWidth);
			pixelX = pixelNum / destHeight;
			pixelY = pixelNum % destHeight;
			gridStartX = (pixelX + 1 - (s32)kernelWidth) <= 0 ? 0 :
					(pixelX + 1 - (s32)kernelWidth);
			gridStartY = (pixelY + 1 - (s32)kernelHeight) <= 0 ? 0 :
					(pixelY + 1 - (s32)kernelHeight);
			for(s32 j=gridStartX; (j<=pixelX) && ((j + kernelWidth) <= destWidth); j++) {
				for(s32 k=gridStartY; (k<=pixelY) && ((k + kernelHeight) <= destHeight) ; k++) {
					// (Cx, Cy) = (j + kernelMiddleX, k + kernelMiddleY) are coordinates of center pixel in grid
					// (Rx, Ry) = (Cx - pixelX, Cy - pixelY) gives coordinates of pixel in refernce
					// to center pixel, such that center pixel of grid is mapped is mapped to (0,0)
					neighNum = (pixelX - j) * kernelHeight + (pixelY - k);
					//(j * sourceHeight + k) is pixel number of center of grid in source
					//i.e result of convolution
					this->at(i,img) += source.at((j * sourceHeight + k) * destChannels * kernelWidth * kernelHeight +
							ch * kernelWidth * kernelHeight + neighNum, img);
							//source.at(ch * destHeight * destWidth + pixelNum * kernelWidth * kernelHeight + neighNum, img);
				}
			}
		}
	}
}

template<typename T>
void Matrix<T>::prepareConvolutionSame(const Matrix<T>& source, const u32 sourceWidth, const u32 sourceHeight,
		const u32 sourceChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY)
{
	require_eq(nColumns_, source.nColumns_);

	require_gt(strideX, 0);
	require_gt(strideY, 0);
	require_gt(sourceWidth, strideX);
	require_gt(sourceHeight, strideY);

	require_eq(source.nRows_, sourceChannels * sourceWidth * sourceHeight);

	require_eq(kernelHeight % 2, 1);
	require_eq(kernelWidth % 2, 1);

	u32 destWidth = (u32)ceil((f32)sourceWidth / (f32)strideX);
	u32 destHeight = (u32)ceil((f32)sourceHeight / (f32)strideY);

	require_eq(nRows_, sourceChannels * destWidth * destHeight * kernelWidth * kernelHeight);

	s32 pixelNum, channelNum, neighbNum, pixelX, pixelY, neighbX, neighbY;
	s32 kernelMiddleX = kernelWidth / 2;
	s32 kernelMiddleY = kernelHeight / 2;

	for(u32 img=0; img<nColumns_; img++) {
		for(u32 j = 0; j < nRows_; j++) {
			pixelNum = j / (kernelHeight * kernelWidth * sourceChannels);
			//calculates the (x,y) of source pixel to which this neighborhood corresponds to
			pixelX = (pixelNum / destHeight) * strideX;
			pixelY = (pixelNum % destHeight) * strideY;

			//calculates the channel in source along with neighborhood number
			channelNum = j % (kernelHeight * kernelWidth * sourceChannels);
			neighbNum = channelNum % (kernelHeight * kernelWidth);
			channelNum = channelNum / (kernelWidth * kernelHeight);

			//calculates the (x,y) component of neighboring pixel w.r.t current pixel
			neighbX = (neighbNum / kernelHeight) - kernelMiddleX;
			neighbY = (neighbNum % kernelHeight) - kernelMiddleY;

			this->at(j,img) = ((pixelX + neighbX) < 0 || (pixelX + neighbX) >= sourceWidth ||
					(pixelY + neighbY) < 0 || (pixelY + neighbY) >= sourceHeight) ? 0 :
							source.at(channelNum * sourceWidth * sourceHeight +
									(pixelX + neighbX) * sourceHeight + (pixelY + neighbY),img);
		}
	}
}
template<typename T>
void Matrix<T>::prepareConvolutionSameBackProp(const Matrix<T>& source, const u32 destWidth, const u32 destHeight,
		const u32 destChannels, const u32 kernelWidth, const u32 kernelHeight, const u32 strideX, const u32 strideY)
{
	require_eq(nColumns_, source.nColumns_);
	require_gt(strideX, 0);
	require_gt(strideY, 0);
	require_gt(destWidth, strideX);
	require_gt(destHeight, strideY);
	require_eq(nRows_, destWidth * destHeight * destChannels);
	require_eq(kernelWidth % 2, 1);
	require_eq(kernelHeight % 2, 1);


	require_eq(source.nRows_, (destWidth / strideX) * (destHeight / strideY) * destChannels * kernelWidth * kernelHeight);

	s32 gridStartX, gridStartY, neighNum, centerPixel;

	s32 kernelMiddleX = (s32)kernelWidth/2;
	s32 kernelMiddleY = (s32)kernelHeight/2;

	for(s32 img = 0; img<nColumns_; img++) {
		for(s32 ch=0; ch<destChannels; ch++) {
			for(s32 x=0; x<destWidth; x++) {
				for(s32 y=0; y<destHeight; y++) {
					this->at(ch * destHeight * destWidth + x * destHeight + y, img) = 0;

					gridStartX = (x + 1 - (s32)kernelWidth) <= (-1 * kernelMiddleX) ? (-1 * kernelMiddleX) :
							(x + 1 - (s32)kernelWidth);
					gridStartY = (y + 1 - (s32)kernelHeight) <= (-1 * kernelMiddleY) ? (-1 * kernelMiddleY) :
							(y + 1 - (s32)kernelHeight);

					for(s32 gridX = gridStartX; (gridX <= x) && ((gridX + kernelMiddleX)<destWidth); gridX++) {
						//else the pixel hasn't participated in convolution
						if ((gridX + kernelMiddleX) % strideX == 0) {
							for(s32 gridY = gridStartY; (gridY <= y) && ((gridY + kernelMiddleY)<destHeight); gridY++) {
								//else the pixel hasn't participated in convolution
								if ((gridY + kernelMiddleY) % strideY == 0) {
									neighNum = (x - gridX) * kernelHeight + (y - gridY);
									//ID of main pixel of neighborhood in the result of convolution
									centerPixel = (((gridX + kernelMiddleX) / strideX) * (destHeight / strideY)) + (gridY + kernelMiddleY) / strideY;
									this->at(ch * destHeight * destWidth + x * destHeight + y, img) +=
											source.at(centerPixel * destChannels * kernelWidth * kernelHeight
													+ ch * kernelHeight * kernelWidth + neighNum,img);
								}
							}
						}
					}
				}
			}
		}
	}
}
template<typename T>
void Matrix<T>::rearrange(const Matrix<T>& source, const u32 numImages)
{
	//Source has size MxN
	//Where M is assumed to be NumChannelsInDest
	//Where N is assumed to be (NumPixelsInDest x NumImagesInBatch)
	//Dest is assumed to be KxL
	//Where K is (NumPixelsInDest x NumChannelsInDest)
	//Where L is assumed to be NumImagesInBatch
	require_eq(source.nColumns_ % numImages , 0);
	u32 numPixels = source.nColumns_ / numImages;
	require_eq(nColumns_, numImages);
	require_eq(nRows_, numPixels * source.nRows_);
	for(u32 img = 0; img < nColumns_; img++) {
		for(u32 i=0; i<nRows_; i++) {
			u32 ch = i / numPixels;
			u32 pix = i % numPixels;
			this->at(i, img) = source.at(ch, img * numPixels + pix);
		}
	}
}
template<typename T>
void Matrix<T>::rearrangeBackProp(const Matrix<T>& source, const u32 channels)
{
	//Source is MxN
	//Where N is assumed to be number of images in batch
	//Where M is assumed to be (NumPixels x channels)
	require_eq(source.nRows_ % channels, 0);
	u32 numPixels = source.nRows_ / channels;
	require_eq(nRows_, channels);
	require_eq(nColumns_, numPixels * source.nColumns_);
	for(u32 i=0; i<nColumns_; i++) {
		u32 img = i / numPixels;
		u32 pix = i % numPixels;
		for(u32 j=0; j<nRows_; j++) {
			this->at(j, i) = source.at(j*numPixels + pix ,img);
		}
	}
}
template<typename T>
void Matrix<T>::fill(u32 rowA, u32 columnA, u32 rowB, u32 columnB, T value) {
	require(!needsU64Space_);
	require_lt(rowA, nRows_);
	require_lt(rowB, nRows_);
	require_lt(columnA, nColumns_);
	require_lt(columnB, nColumns_);
	if ( (columnA < columnB) || ((columnA == columnB) && (rowA < rowB)) )
		std::fill(elem_ + columnA * nRows_ + rowA, elem_ + columnB * nRows_ + rowB + 1, value);
}

template<typename T>
void Matrix<T>::ensureMinimalValue(const T threshold) {
	require(!needsU64Space_);
	for (u32 row = 0; row < nRows_; row++) {
		for (u32 column = 0; column < nColumns_; column++) {
			if (at(row, column) < threshold)
				at(row, column) = threshold;
		}
	}
}

template<typename T>
void Matrix<T>::ensureMaximalValue(const T threshold) {
	require(!needsU64Space_);
	for (u32 row = 0; row < nRows_; row++) {
		for (u32 column = 0; column < nColumns_; column++) {
			if (at(row, column) > threshold)
				at(row, column) = threshold;
		}
	}
}

template<typename T>
std::string Matrix<T>::toString(bool transpose) const {
	require(!needsU64Space_);
	require(nRows_ > 0);
	require(nColumns_ > 0);
	std::stringstream s;
	if (transpose) {
		for (u32 i = 0; i < nColumns_; i++) {
			for (u32 j = 0; j < nRows_; j++) {
				s << at(j,i);
				if (j < nRows_ - 1) s << " ";
			}
			if (i != nColumns_ - 1)
				s << std::endl;
		}
	}
	else {
		for (u32 i = 0; i < nRows_; i++) {
			for (u32 j = 0; j < nColumns_; j++) {
				s << at(i,j);
				if (j < nColumns_ - 1) s << " ";
			}
			if (i != nRows_ - 1)
				s << std::endl;
		}
	}
	return s.str();
}

template<typename T>
void Matrix<T>::tanh(){
	require(!needsU64Space_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] = std::tanh(elem_[i]);
}

template<typename T>
void Matrix<T>::exp(){
	mt_vr_exp(nRows_ * nColumns_, elem_ , elem_, nThreads_);
}

template<typename T>
void Matrix<T>::signedPow(T p){
	require(!needsU64Space_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		if (elem_[i] < 0)
			elem_[i] = -std::pow(-elem_[i], p);
		else
			elem_[i] = std::pow(elem_[i], p);
	}
}

template<typename T>
void Matrix<T>::log(){
	require(!needsU64Space_);
	vr_log(nRows_ * nColumns_, elem_ , elem_);
}

template<typename T>
void Matrix<T>::sin(){
	require(!needsU64Space_);
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		elem_[i] = std::sin(elem_[i]);
	}
}

template<typename T>
void Matrix<T>::cos(){
	require(!needsU64Space_);
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		elem_[i] = std::cos(elem_[i]);
	}
}

template<typename T>
void Matrix<T>::asin(){
	require(!needsU64Space_);
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		elem_[i] = std::asin(elem_[i]);
	}
}

template<typename T>
void Matrix<T>::acos(){
	require(!needsU64Space_);
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		elem_[i] = std::acos(elem_[i]);
	}
}

template<typename T>
T Matrix<T>::maxValue() const {
	require(!needsU64Space_);
	Vector<T> tmp(nColumns_);
	tmp.getMaxOfColumns(*this);
	return tmp.max();
}

template<typename T>
void Matrix<T>::abs(){
	require(!needsU64Space_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] = std::abs(elem_[i]);
}

template<typename T>
void Matrix<T>::sigmoid(T gamma){
	require(!needsU64Space_);
	scale(-gamma);
	exp();
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] = 1.0 / (1.0 + elem_[i]);
}

template<typename T>
void Matrix<T>::triangle(){
	require(!needsU64Space_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		if (std::abs(elem_[i]) > 1)
			elem_[i] = 0;
		else
			elem_[i] = 1.0 - std::abs(elem_[i]);
	}
}

template<typename T>
void Matrix<T>::softmax(){
	require(!needsU64Space_);
	// softmax: t(i) = exp(s(i)               ) / sum_j { exp(s(j)                ) } (column-wise)
	// softmax: t(i) = exp(s(i) - MAX_j{s(j)} ) / sum_k { exp(s(k) - max_j {s(j)} ) }, more robust computation (avoids overflow)

	// get max of all columns and remove it from all rows
	Vector<T> tmp(nColumns_);
	tmp.getMaxOfColumns(*this);
	addToAllRows(tmp, (T) -1.0);
	// exponentiate
	exp();
	// accumulate entries of each column
	tmp.setToZero();
	tmp.addSummedRows(*this);

	// compute actual softmax output for each column
	divideColumnsByScalars(tmp);
}

template<typename T>
T Matrix<T>::sum() const {
	require(!needsU64Space_);
	T result = 0;
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		result += elem_[i];
	}
	return result;
}

template<typename T>
void Matrix<T>::max() {
	require(!needsU64Space_);
	// get index of max in each column
	Vector<T> tmp(nColumns_);
	argMax(tmp);
	setToZero();
#pragma omp parallel for
	for (u32 column = 0; column < nColumns_; column++) {
		at(tmp.at(column), column) = 1.0;
	}
}

template<typename T>
void Matrix<T>::max(const Matrix<T> &A, const Matrix<T> &B) {
	require(!needsU64Space_);
	require_eq(A.nRows_, B.nRows_);
	require_eq(A.nColumns_, B.nColumns_);
	require_eq(A.nRows_, nRows_);
	require_eq(A.nColumns_, nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		elem_[i] = std::max(A.elem_[i], B.elem_[i]);
	}
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithKroneckerDelta(const Matrix<T> &A, const Matrix<T> &B) {
	require(!needsU64Space_);
	require_eq(A.nRows_, B.nRows_);
	require_eq(A.nColumns_, B.nColumns_);
	require_eq(A.nRows_, nRows_);
	require_eq(A.nColumns_, nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		if (A.elem_[i] != B.elem_[i])
			elem_[i] = 0;
	}
}

template<typename T>
void Matrix<T>::clone(const Matrix<T> &X, u32 nClones) {
	require(!needsU64Space_);
	require_eq(X.nRows_ * nClones, nRows_);
	require_eq(X.nColumns_, nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < X.nRows_; i++) {
		for (u32 j = 0; j < X.nColumns_; j++) {
			for (u32 n = 0; n < nClones; n++) {
				at(n * X.nRows_ + i, j) = X.at(i, j);
			}
		}
	}
}

template<typename T>
void Matrix<T>::cloneElementwise(const Matrix<T> &X, u32 nClones) {
	require(!needsU64Space_);
	require_eq(X.nRows_ * nClones, nRows_);
	require_eq(X.nColumns_, nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < X.nRows_ * X.nColumns_; i++) {
		for (u32 n = 0; n < nClones; n++) {
			elem_[i * nClones + n] = X.elem_[i];
		}
	}
}

template<typename T>
void Matrix<T>::addElementsByModuloIndex(const Matrix<T>& X) {
	require(!needsU64Space_);
	require_eq(X.nRows() % nRows_, 0);
	require_eq(X.nColumns(), nColumns_);
#pragma omp parallel for
	for (u32 column = 0; column < X.nColumns(); column++) {
		for (u32 row = 0; row < X.nRows(); row++) {
			at(row % nRows_, column) += X.at(row, column);
		}
	}
}

template<typename T>
void Matrix<T>::chiSquareFeatureMap(const Matrix<T>& X, u32 n, T samplingDistance) {
	require(!needsU64Space_);
	require_eq(X.nColumns(), nColumns_);
	require_eq(X.nRows() * (2*n + 1), nRows_);

	u32 J = 2 * n + 1; // compute J elements j = 0,...,J-1 of the feature map for each feature in X
#pragma omp parallel for
	for (u32 column = 0; column < X.nColumns(); column++) {
		for (u32 row = 0; row < X.nRows(); row++) {
			T x = std::max(X.at(row, column), Types::absMin<T>());
			// compute J = 2*n + 1 elements from X.at(row, column)
			for (u32 j = 0; j < J; j++) {
				if (j == 0) { // j = 0
					at(row * J + j, column) = std::sqrt(samplingDistance * x);
				}
				else if (j % 2 == 1) { // j > 0 odd
					T kappa = 1.0 / std::cosh(M_PI * (j+1)/2 * samplingDistance);
					at(row * J + j, column) = std::sqrt(2 * kappa * samplingDistance * x) *
							std::cos((j+1)/2 * samplingDistance * std::log(x));
				}
				else { // j > 0 even
					T kappa = 1.0 / std::cosh(M_PI * j/2 * samplingDistance);
					at(row * J + j, column) = std::sqrt(2 * kappa * samplingDistance * x) *
							std::sin(j/2 * samplingDistance * std::log(x));
				}
			}
		}
	}
}

template<typename T>
void Matrix<T>::histogramIntersectionFeatureMap(const Matrix<T>& X, u32 n, T samplingDistance) {
	require(!needsU64Space_);
	require_eq(X.nColumns(), nColumns_);
	require_eq(X.nRows() * (2*n + 1), nRows_);

	u32 J = 2 * n + 1; // compute J elements j = 0,...,J-1 of the feature map for each feature in X
#pragma omp parallel for
	for (u32 column = 0; column < X.nColumns(); column++) {
		for (u32 row = 0; row < X.nRows(); row++) {
			T x = std::max(X.at(row, column), Types::absMin<T>());
			// compute J = 2*n + 1 elements from X.at(row, column)
			for (u32 j = 0; j < J; j++) {
				if (j == 0) { // j = 0
					at(row * J + j, column) = std::sqrt(2 / M_PI * samplingDistance * x);
				}
				else if (j % 2 == 1) { // j > 0 odd
					T kappa = 2.0 / (M_PI * (1 + 4 * (j+1)/2 * samplingDistance * (j+1)/2 * samplingDistance));
					at(row * J + j, column) = std::sqrt(2 * kappa * samplingDistance * x) *
							std::cos((j+1)/2 * samplingDistance * std::log(x));
				}
				else { // j > 0 even
					T kappa = 2.0 / (M_PI * (1 + 4 * j/2 * samplingDistance * j/2 * samplingDistance));
					at(row * J + j, column) = std::sqrt(2 * kappa * samplingDistance * x) *
							std::sin(j/2 * samplingDistance * std::log(x));
				}
			}
		}
	}
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithApproximateFeatureMapDerivative(const Matrix<T>& X, u32 n, T samplingDistance, T kappa0) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	require(nRows_ % (2*n + 1) == 0);
	// loop over all elements
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		u32 j = i % (2 * n + 1);
		if (j == 0) {
			elem_[i] *= X.elem_[i];
		}
		else if (j % 2 == 1) {
			elem_[i] *= X.elem_[i] - (j+1) * samplingDistance * X.elem_[i + 1];
		}
		else {
			elem_[i] *= X.elem_[i] + j * samplingDistance * X.elem_[i - 1];
		}
		elem_[i] *= kappa0 * samplingDistance / (2.0 * X.elem_[i - j] * X.elem_[i - j]);
	}
}

template<typename T>
u32 Matrix<T>::argAbsMin(u32 column) const {
	require(!needsU64Space_);
	require_lt(column, nColumns_);
	return Math::iamin(nRows_, elem_ + column * nRows_, 1);
}

template<typename T>
u32 Matrix<T>::argAbsMax(u32 column) const {
	require(!needsU64Space_);
	require_lt(column, nColumns_);
	return Math::iamax(nRows_, elem_ + column * nRows_, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::argMax(Vector<S>& v) const {
	require(!needsU64Space_);
	require_eq(v.nRows(), nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++) {
		T maxVal = at(0, i);
		v.at(i) = 0;
		for (u32 j = 1; j < nRows_; j++){
			if (at(j, i) > maxVal){
				maxVal = at(j, i);
				v.at(i) = j;
			}
		}
	}
}

template<typename T>
template<typename S>
void Matrix<T>::add(const Matrix<S> &X, S alpha){
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	Math::axpy<S,T>(nRows_ * nColumns_, alpha, X.elem_, 1, elem_, 1);
}

template<typename T>
void Matrix<T>::addWeighted(const Matrix<T>& X, const Vector<T>& weights) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_le(X.nColumns(), nColumns_);
	require_eq(X.nColumns(), weights.nRows());
#pragma omp parallel for
	for (u32 row = 0; row < nRows_; row++) {
		for (u32 col = 0; col < X.nColumns(); col++) {
			at(row, col) += weights.at(col) * X.at(row, col);
		}
	}
}

template<typename T>
T Matrix<T>::dot(const Matrix<T> &X) const {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	return Math::mt_dot(nRows_ * nColumns_, elem_, X.elem_, nThreads_);
}

template<typename T>
void Matrix<T>::scale(T alpha){
	require(!needsU64Space_);
	Math::mt_scal(nRows_ * nColumns_, alpha, elem_, nThreads_);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const Matrix<S> &X){
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	Math::copy<S,T>(nRows_ * nColumns_, X.elem_, 1, elem_, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const S *X, u32 rowOffset, u32 colOffset){
	require(!needsU64Space_);
	require_lt(rowOffset, nRows_);
	require_lt(colOffset, nColumns_);
	return Math::copy<S,T>(nRows_ * nColumns_ - colOffset * nRows_ - rowOffset, X, 1, elem_ + colOffset * nRows_ + rowOffset, 1);
}

template<typename T>
template<typename S>
void Matrix<T>::copy(const std::vector<S> &X, u32 rowOffset, u32 colOffset){
	require(!needsU64Space_);
	require_lt(rowOffset, nRows_);
	require_lt(colOffset, nColumns_);
	return Math::copy<S,T>(X.size(), &X.at(0), 1, elem_ + colOffset * nRows_ + rowOffset, 1);
}

template<typename T>
void Matrix<T>::elementwiseMultiplication(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	// TODO parallelize
	std::transform(elem_, elem_ + nRows_ * nColumns_, X.elem_, elem_, std::multiplies<T>());
}

template<typename T>
void Matrix<T>::rpropUpdate(const Matrix<T> &newGradients, Matrix<T> &oldGradients, Matrix<T> &updateValues,
		const T increasingFactor, const T decreasingFactor, const T maxUpdateValue, const T minUpdateValue) {
	require(!needsU64Space_);
	require_eq(oldGradients.nRows(), nRows_);
	require_eq(newGradients.nRows(), nRows_);
	require_eq(oldGradients.nColumns(), nColumns_);
	require_eq(newGradients.nColumns(), nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		for (u32 j = 0; j < nColumns_; j++) {
			T change = oldGradients.at(i, j) *  newGradients.at(i, j);
			if (change > 0) {
				updateValues.at(i, j) *= increasingFactor;
				if (updateValues.at(i, j) > maxUpdateValue)
					updateValues.at(i, j) = maxUpdateValue;
			} else if (change < 0) {
				updateValues.at(i, j) *= decreasingFactor;
				if (updateValues.at(i, j) < minUpdateValue)
					updateValues.at(i, j) = minUpdateValue;
			}
			if (newGradients.at(i, j) > 0)
				at(i, j) += -updateValues.at(i, j);
			else if (newGradients.at(i, j) < 0)
				at(i, j) += updateValues.at(i, j);
			oldGradients.at(i, j) = newGradients.at(i, j);
		}
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithSigmoidDerivative(const Matrix<T> &X) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] *= X.elem_[i] * (1.0 - X.elem_[i]);
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithTriangleDerivative(const Matrix<T> &X) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
#pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		if ((std::abs(X.elem_[i]) > 1) || (X.elem_[i] == 0))
			elem_[i] = 0;
		else if (X.elem_[i] > 0) // range (0,1] has derivative -1
			elem_[i] *= -1;
	}
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithTanhDerivative(const Matrix<T> &X) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] *= 1.0 - X.elem_[i] * X.elem_[i];
}

template<typename T>
void Matrix<T>::multiplicationWithSoftmaxDerivative(const Math::Matrix<T>& softmax) {
	require(!needsU64Space_);
	require_eq(softmax.nRows(), nRows_);
	require_eq(softmax.nColumns(), nColumns_);
	Vector<T> v(nColumns_);
	v.columnwiseInnerProduct(softmax, *this);
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			at(row,column) = softmax.at(row,column) * (at(row,column) -  v.at(column));
		}
	}
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithClippedDerivative(const Math::Matrix<T>& X, T thresholdLeft, T thresholdRight) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		if ((X.elem_[i] <= thresholdLeft) || (X.elem_[i] >= thresholdRight)) elem_[i] = 0;
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithLogDerivative(const Math::Matrix<T>& X) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++)
		elem_[i] *= std::exp(-X.elem_[i]);
}

template<typename T>
void Matrix<T>::elementwiseMultiplicationWithSignedPowDerivative(const Math::Matrix<T>& X, T p) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
# pragma omp parallel for
	for (u32 i = 0; i < nRows_ * nColumns_; i++) {
		if (X.elem_[i] == 0)
			elem_[i] = 0;
		else if (X.elem_[i] < 0)
			elem_[i] *= p * std::pow(-X.elem_[i], p - 1);
		else
			elem_[i] *= p * std::pow(X.elem_[i], p - 1);
	}
}

template<typename T>
void Matrix<T>::multiplicationWithL2NormalizationDerivative(const Math::Matrix<T>& X, const Math::Vector<T>& norm) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	require_eq(norm.nRows(), nColumns_);
	Vector<T> v(nColumns_);
	v.columnwiseInnerProduct(X, *this);
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			at(row,column) = (at(row,column) -  X.at(row, column) * v.at(column)) / norm.at(column);
		}
	}
}

template<typename T>
void Matrix<T>::addSummedNeighborsInARow(const Matrix<T>& X, u32 nNeighbors) {
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_ * nNeighbors);
	require_eq(X.nColumns(), nColumns_);
#pragma omp parallel for
	for (u32 row = 0; row < nRows_; row++) {
		for (u32 column = 0; column < nColumns_; column++) {
			for (u32 n = 0; n < nNeighbors; n++) {
				at(row, column) += X.at(row * nNeighbors + n, column);
			}
		}
	}
}

template<typename T>
u32 Matrix<T>::nClassificationErrors(const Matrix<T>& targets) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	Vector<T> tmp(nColumns_);
	argMax(tmp);
	u32 nErrors = 0;
	for (u32 col = 0; col < nColumns_; col++) {
		if (targets.at(tmp.at(col), col) != 1.0) {
			nErrors++;
		}
	}
	return nErrors;
}


template<typename T>
T Matrix<T>::crossEntropyObjectiveFunction(const Matrix<T>& targets) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			if (targets.at(row, column) == 1.0)
				objFctn += -std::log(at(row, column));
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::weightedCrossEntropyObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	require_eq(nColumns_, weights.nRows());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			if (targets.at(row, column) == 1.0)
				objFctn += -std::log(at(row, column)) * weights[column];
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::squaredErrorObjectiveFunction(const Matrix<T>& targets) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			objFctn += (at(row, column) - targets.at(row, column)) * (at(row, column) - targets.at(row, column));
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::weightedSquaredErrorObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	require_eq(nColumns_, weights.nRows());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			objFctn += (at(row, column) - targets.at(row, column)) * (at(row, column) - targets.at(row, column)) * weights[column];
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::smoothedL1ObjectiveFunction(const Matrix<T>& targets) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			if (std::abs(at(row, column) - targets.at(row, column)) <= 1.0)
				objFctn += 0.5 * (at(row, column) - targets.at(row, column)) * (at(row, column) - targets.at(row, column));
			else
				objFctn += std::abs((at(row, column) - targets.at(row, column))) - 0.5;
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::weightedSmoothedL1ObjectiveFunction(const Matrix<T>& targets, const Vector<T>& weights) const {
	require(!needsU64Space_);
	require_eq(nRows_, targets.nRows());
	require_eq(nColumns_, targets.nColumns());
	require_eq(nColumns_, weights.nRows());
	T objFctn = 0;
#pragma omp parallel for reduction(+:objFctn)
	for (u32 column = 0; column < nColumns_; column++) {
		for (u32 row = 0; row < nRows_; row++) {
			if (std::abs(at(row, column) - targets.at(row, column)) <= 1.0)
				objFctn += 0.5 * (at(row, column) - targets.at(row, column)) * (at(row, column) - targets.at(row, column)) * weights.at(column);
			else
				objFctn += (std::abs((at(row, column) - targets.at(row, column))) - 0.5) * weights.at(column);
		}
	}
	return objFctn;
}

template<typename T>
T Matrix<T>::dotWithColumn(const Vector<T> &v, u32 thisColumnIndex) const {
	require(!needsU64Space_);
	require_eq(v.nRows(), nRows_);
	require_lt(thisColumnIndex, nColumns_);
	return Math::dot(nRows_, &(at(0, thisColumnIndex)), 1, v.begin(), 1);
}

template<typename T>
void Matrix<T>::setToSecondOrderFeatures(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(nRows_, X.nRows_ + (X.nRows_ * (X.nRows_ + 1)) / 2);
	// copy first order features
	copyBlockFromMatrix(X, 0, 0, 0, 0, X.nRows_, X.nColumns_);
	// append second order features
	appendSecondOrderFeatures(X, X.nRows_);
}

template<typename T>
void Matrix<T>::setToDiagonalSecondOrderFeatures(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(nRows_, X.nRows_ * 2);
	// copy first order features
	copyBlockFromMatrix(X, 0, 0, 0, 0, X.nRows_, X.nColumns_);
	// append diagonal second order features
	appendDiagonalSecondOrderFeatures(X, X.nRows_);
}

template<typename T>
void Matrix<T>::setToThirdOrderFeatures(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(nRows_, X.nRows_ + (X.nRows_ * (X.nRows_ + 1)) / 2 + (X.nRows_ * (X.nRows_ + 1) * (X.nRows_ + 2)) / 6);
	// copy first order features
	copyBlockFromMatrix(X, 0, 0, 0, 0, X.nRows_, X.nColumns_);
	// append second order features
	appendSecondOrderFeatures(X, X.nRows_);
	// append second order features
	appendThirdOrderFeatures(X, X.nRows_ + (X.nRows_ * (X.nRows_ + 1)) / 2);
}

template<typename T>
void Matrix<T>::setToDiagonalThirdOrderFeatures(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(nRows_, X.nRows_ * 3);
	// copy first order features
	copyBlockFromMatrix(X, 0, 0, 0, 0, X.nRows_, X.nColumns_);
	// append second order features
	appendDiagonalSecondOrderFeatures(X, X.nRows_);
	// append second order features
	appendDiagonalThirdOrderFeatures(X, X.nRows_ * 2);
}

template<typename T>
void Matrix<T>::gaussianMixturePosteriors(const Matrix<T> &X, const Matrix<T> &means, const Matrix<T> &variances, const Vector<T> &weights) {
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(X.nRows_, means.nColumns_);
	require_eq(X.nRows_, variances.nColumns_);
	require_eq(means.nRows_, weights.nRows_);
	require_eq(means.nRows_, variances.nRows_);
	require_eq(nRows_, means.nRows_);
#pragma omp parallel for
	for (u32 n = 0; n < nColumns_; n++) {
		for (u32 i = 0; i < nRows_; i++) {
			Float expn = 0;
			Float det = 0;
			for (u32 d = 0; d < X.nRows_; d++) {
				expn += (X.at(d, n) - means.at(i, d)) * (X.at(d, n) - means.at(i, d)) / variances.at(i, d);
				det += std::log(variances.at(i, d));
			}
			at(i, n) = std::log(weights.at(i)) - 0.5 * expn - 0.5 * std::log(2 * M_PI) * X.nRows_ - 0.5 * det;
		}
	}
	softmax();
}

template<typename T>
void Matrix<T>::fisherEncoding(const Matrix<T> &X, const Matrix<T> &means, const Matrix<T> &variances, const Vector<T> &weights) {
	require(!needsU64Space_);
	require_eq(nColumns_, X.nColumns_);
	require_eq(X.nRows_, means.nColumns_);
	require_eq(X.nRows_, variances.nColumns_);
	require_eq(means.nRows_, weights.nRows_);
	require_eq(means.nRows_, variances.nRows_);
	require_eq(nRows_, X.nRows_ * means.nRows_ * 2);

	Matrix gamma(means.nRows_, nColumns_);
	gamma.gaussianMixturePosteriors(X, means, variances, weights);

	for (u32 n = 0; n < nColumns_; n++) {
		for (u32 i = 0; i < means.nRows_; i++) {
			for (u32 d = 0; d < X.nRows_; d++) {
				at(d + i*X.nRows_, n) = gamma.at(i, n) * (X.at(d, n) - means.at(i, d))
						/ (std::sqrt(variances.at(i, d) * weights.at(i)));
				at(d + i*X.nRows_ + X.nRows_ * means.nRows_, n) = gamma.at(i, n)
						* ((X.at(d, n) - means.at(i, d)) * (X.at(d, n) - means.at(i, d)) / variances.at(i, d) - 1.0)
						/ std::sqrt(2*weights.at(i));
			}
		}
	}
}

template<typename T>
void Matrix<T>::dropout(const T dropoutProbability) {
	require(!needsU64Space_);
	for (u32 row = 0; row < nRows_; row++) {
		for (u32 column = 0; column < nColumns_; column++) {
			if (Math::Random::random() < dropoutProbability)
				at(row, column) = 0.0;
		}
	}
}

template<typename T>
void Matrix<T>::appendSecondOrderFeatures(const Matrix<T> &X, u32 offset){
	require(!needsU64Space_);
	for (u32 column = 0; column < nColumns_; column++){
		u32 pos = offset;
		for (u32 i = 0; i < X.nRows_; ++ i) {
			for (u32 j = i; j < X.nRows_; ++ j) {
				at(pos, column) = X.at(i, column) * X.at(j, column);
				pos++;
			}
		}
	}
}

template<typename T>
void Matrix<T>::appendDiagonalSecondOrderFeatures(const Matrix<T> &X, u32 offset){
	require(!needsU64Space_);
	for (u32 column = 0; column < nColumns_; column++){
		u32 pos = offset;
		for (u32 i = 0; i < X.nRows_; ++ i) {
			at(pos, column) = X.at(i, column) * X.at(i, column);
			pos++;
		}
	}
}

template<typename T>
void Matrix<T>::appendThirdOrderFeatures(const Matrix<T> &X, u32 offset){
	require(!needsU64Space_);
	for (u32 column = 0; column < nColumns_; column++){
		u32 pos = offset;
		for (u32 i = 0; i < X.nRows_; ++ i) {
			for (u32 j = i; j < X.nRows_; ++ j) {
				for (u32 k = j; k < X.nRows_; ++ k) {
					at(pos, column) = X.at(i, column) * X.at(j, column) * X.at(k, column);
					pos++;
				}
			}
		}
	}
}

template<typename T>
void Matrix<T>::appendDiagonalThirdOrderFeatures(const Matrix<T> &X, u32 offset){
	require(!needsU64Space_);
	for (u32 column = 0; column < nColumns_; column++){
		u32 pos = offset;
		for (u32 i = 0; i < X.nRows_; ++ i) {
			at(pos, column) = X.at(i, column) * X.at(i, column) * X.at(i, column);
			pos++;
		}
	}
}

template<typename T>
void Matrix<T>::elementwiseDivision(const Matrix<T> &X){
	require(!needsU64Space_);
	require_eq(X.nRows(), nRows_);
	require_eq(X.nColumns(), nColumns_);
	std::transform(elem_, elem_ + nRows_ * nColumns_, X.elem_, elem_, std::divides<T>());
}

template<typename T>
void Matrix<T>::addConstantElementwise(T c) {
	require(!needsU64Space_);
	std::transform(elem_, elem_ + nRows_ * nColumns_, elem_, std::bind2nd(std::plus<T>(), c));
}

template<typename T>
void Matrix<T>::addToColumn(const Vector<T> &v, u32 column, T alpha) {
	require(!needsU64Space_);
	require_lt(column, nColumns_);
	require_eq(v.nRows(), nRows_);
	Math::mt_axpy(nRows_, alpha, v.begin(), elem_ + column * nRows_, nThreads_);
}

template<typename T>
void Matrix<T>::multiplyColumnByScalar(u32 column, T alpha) {
	require(!needsU64Space_);
	require_lt(column, nColumns_);
	Math::scal(nRows_, alpha, &at(0, column), 1);
}

template<typename T>
void Matrix<T>::multiplyRowByScalar(u32 row, T alpha) {
	require(!needsU64Space_);
	require_lt(row, nRows_);
	Math::scal(nColumns_, alpha, &at(row, 0), nRows_);
}

template<typename T>
void Matrix<T>::addToRow(const Vector<T> &v, u32 row, T alpha) {
	require(!needsU64Space_);
	require_lt(row, nRows_);
	require_eq(v.nRows(), nColumns_);
	Math::axpy(nColumns_, alpha, v.begin(), 1, elem_ + row, nRows_);
}

template<typename T>
void Matrix<T>::addToAllColumns(const Vector<T> &v, T alpha){
	require(!needsU64Space_);
	require_eq(v.nRows(), nRows_);
# pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::mt_axpy(nRows_, alpha, v.begin(), elem_ + i*nRows_, nThreads_);
}
template<typename T>
void Matrix<T>::addToAllChannels(const Vector<T> &v, const u32 channels, T alpha)
{
	require(!needsU64Space_);
	require_eq(v.nRows(), channels);
	require_eq(nRows_ % channels, 0);

	u32 channelSize = nRows_ / channels;
//#pragma omp parallel for
	for(u32 j=0; j<nColumns_; j++) {
		for(u32 i=0; i<nRows_; i++) {
			this->at(i,j) += alpha*v.at(i/channelSize);
		}
	}
}
template<typename T>
void Matrix<T>::addToAllRows(const Vector<T> &v, T alpha){
	require(!needsU64Space_);
	require_eq(v.nRows(), nColumns_);
#pragma omp parallel for
	for (u32 j = 0; j < nColumns_; j++){
		T value = alpha * v.at(j);
		std::transform(elem_ + j*nRows_, elem_ + (j+1)*nRows_, elem_ + j*nRows_, std::bind2nd(std::plus<T>(), value));
	}
}

template<typename T>
void Matrix<T>::multiplyColumnsByScalars(const Vector<T> &scalars){
	require(!needsU64Space_);
	require_eq(nColumns_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::scal(nRows_, scalars[i], &at(0, i), 1);
}

template<typename T>
void Matrix<T>::divideColumnsByScalars(const Vector<T> &scalars){
	require(!needsU64Space_);
	require_eq(nColumns_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nColumns_; i++)
		Math::scal(nRows_, (T) 1.0 / scalars[i], &at(0, i), 1);
}

template<typename T>
void Matrix<T>::multiplyRowsByScalars(const Vector<T> &scalars){
	require(!needsU64Space_);
	require_eq(nRows_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		Math::scal(nColumns_, scalars[i], &at(i, 0), nRows_);
}

template<typename T>
void Matrix<T>::divideRowsByScalars(const Vector<T> &scalars){
	require(!needsU64Space_);
	require_eq(nRows_, scalars.size());
#pragma omp parallel for
	for (u32 i = 0; i < nRows_; i++)
		Math::scal(nColumns_, (T) 1.0 / scalars[i], &at(i, 0), nRows_);
}

template<typename T>
void Matrix<T>::getRow(u32 rowIndex, Math::Vector<T> &row) const {
	require(!needsU64Space_);
	require_lt(rowIndex, nRows_);
	row.resize(nColumns_);
	Math::copy(nColumns_, elem_ + rowIndex, nRows_, row.begin(), 1);
}

template<typename T>
void Matrix<T>::getColumn(u32 columnIndex, Math::Vector<T> &column) const {
	require(!needsU64Space_);
	require_lt(columnIndex, nColumns_);
	column.resize(nRows_);
	Math::copy(nRows_, elem_ + columnIndex * nRows_, 1, column.begin(), 1);
}

template<typename T>
void Matrix<T>::setRow(u32 rowIndex, const Math::Vector<T> &row) {
	require(!needsU64Space_);
	require_lt(rowIndex, nRows_);
	require_eq(row.size(), nColumns_);
	Math::copy(nColumns_, row.begin(), 1, elem_ + rowIndex, nRows_);
}

template<typename T>
void Matrix<T>::setColumn(u32 columnIndex, const Math::Vector<T> &column) {
	require(!needsU64Space_);
	require_lt(columnIndex, nColumns_);
	require_eq(column.size(), nRows_);
	Math::copy(nRows_, column.begin(), 1, elem_ + columnIndex * nRows_, 1);
}

template<typename T>
void Matrix<T>::copyBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns) {
	require(!needsU64Space_);
	require_le(thisColIndex + nColumns, nColumns_);
	require_le(thisRowIndex + nRows, nRows_);
	require_le(colIndexX + nColumns, X.nColumns_);
	require_le(rowIndexX + nRows, X.nRows_);
	for (u32 column = 0; column < nColumns; column++){
		const T *posX =  &X.at(rowIndexX, colIndexX  + column);
		T * posThis = &at(thisRowIndex, thisColIndex + column);
		Math::copy(nRows, posX, 1, posThis, 1);
	}
}

template<typename T>
void Matrix<T>::addBlockFromMatrix(const Math::Matrix<T> &X, u32 rowIndexX, u32 colIndexX, u32 thisRowIndex, u32 thisColIndex, u32 nRows, u32 nColumns, T scale) {
	require(!needsU64Space_);
	require_le(thisColIndex + nColumns, nColumns_);
	require_le(thisRowIndex + nRows, nRows_);
	require_le(colIndexX + nColumns, X.nColumns_);
	require_le(rowIndexX + nRows, X.nRows_);
	for (u32 column = 0; column < nColumns; column++){
		const T *posX =  &X.at(rowIndexX, colIndexX  + column);
		T * posThis = &at(thisRowIndex, thisColIndex + column);
		Math::axpy(nRows, scale, posX, 1, posThis, 1);
	}
}

template<typename T>
void Matrix<T>::addOuterProduct(const Math::Vector<T> &x, const Math::Vector<T> &y, T alpha, u32 lda){
	require(!needsU64Space_);
	require_eq(x.size(), nRows_);
	require_eq(y.size(), nColumns_);
	require_le(lda, nRows_);
	if (lda == 0)
		lda = nRows_;
	Math::ger<T>(CblasColMajor, nRows_, nColumns_, alpha, x.begin(), 1, y.begin(), 1, elem_, lda);
}



// c = \alpha a * b + \gamma c
template<typename T>
void Matrix<T>::_gemm(bool transposeA, bool transposeB, u32 M, u32 N, u32 K,
		T scaleA, T* matrixA, u32 lda,
		T* matrixB, u32 ldb,
		T scaleC, T* matrixC, u32 ldc) {
	Math::gemm<T>(CblasColMajor,
			(transposeA ? CblasTrans : CblasNoTrans),
			(transposeB ? CblasTrans : CblasNoTrans),
			M, N, K,
			scaleA, matrixA, lda,
			matrixB, ldb,
			scaleC, matrixC, ldc);
}

// (*this) = (scaleA * matrixA) * matrixB + scaleC * (*this)
template<typename T>
void Matrix<T>::addMatrixProduct(const Matrix<T> &matrixA, const Matrix<T> &matrixB,
		T scaleC, T scaleA, bool transposeA, bool transposeB) {
	require(!needsU64Space_);
	// final matrix (this) must be of size matrixProductNRows x matrixProductNColumns
	u32 matrixProductNRows, matrixProductNColumns;

	// boundary check depends on the configuration
	if ( (! transposeA) && (! transposeB) ) {
		require_eq(matrixA.nColumns(), matrixB.nRows());
		matrixProductNRows = matrixA.nRows();
		matrixProductNColumns = matrixB.nColumns();
	} else if ( (! transposeA) && (transposeB) ) {
		require_eq(matrixA.nColumns(), matrixB.nColumns());
		matrixProductNRows = matrixA.nRows();
		matrixProductNColumns = matrixB.nRows();
	} else if ( (transposeA) && (! transposeB) ) {
		require_eq(matrixA.nRows(), matrixB.nRows());
		matrixProductNRows = matrixA.nColumns();
		matrixProductNColumns = matrixB.nColumns();
	} else if ( (transposeA) && (transposeB) ) {
		require_eq(matrixA.nRows(), matrixB.nColumns());
		matrixProductNRows = matrixA.nColumns();
		matrixProductNColumns = matrixB.nRows();
	}
	require_eq(matrixProductNRows, nRows_);
	require_eq(matrixProductNColumns, nColumns_);

	// multiply the matrices
	// example: A(2,297); B(297,7000); C(2,7000); transposeA=false; transposeB=false; M=2; N=7000; K=297; LDA=2; LDB=7000; LDC=2
	Math::gemm<T>(CblasColMajor,
			(transposeA ? CblasTrans : CblasNoTrans), (transposeB ? CblasTrans : CblasNoTrans),
			matrixProductNRows, matrixProductNColumns, (transposeA ? matrixA.nRows() : matrixA.nColumns()),
			(T) scaleA, matrixA.begin(), matrixA.nRows(),
			matrixB.begin(), matrixB.nRows(),
			(T) scaleC, (*this).begin(), matrixProductNRows);
}

template<typename T>
void Matrix<T>::writeBinaryHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	if (transpose)
		stream << nColumns_ << nRows_;
	else
		stream << nRows_ << nColumns_;
}

template<typename T>
void Matrix<T>::writeAsciiHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	if (transpose)
		stream << nColumns_ << " " << nRows_ << Core::IOStream::endl;
	else
		stream << nRows_ << " " << nColumns_ << Core::IOStream::endl;
}

template<typename T>
void Matrix<T>::writeBinary(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u32 I, J;
	if (transpose) {
		I = nColumns_;
		J = nRows_;
	}
	else {
		I = nRows_;
		J = nColumns_;
	}
	for (u32 i = 0; i < I; i++) {
		for (u32 j = 0; j < J; j++) {
			if (transpose)
				stream << (f32)at(j, i);
			else
				stream << (f32)at(i, j);
		}
	}
}

template<typename T>
void Matrix<T>::writeAscii(Core::IOStream& stream, bool transpose, bool scientific) {
	require(stream.is_open());
	if (scientific)
		stream << Core::IOStream::scientific;
	if (transpose) {
		for (u32 col = 0; col < nColumns_ - 1; col++) {
			for (u32 row = 0; row < nRows_ - 1; row++) {
				stream << at(row, col) << " ";
			}
			stream << at(nRows_ - 1, col) << Core::IOStream::endl;
		}
		for (u32 row = 0; row < nRows_ - 1; row++) {
			stream << at(row, nColumns_ - 1) << " ";
		}
		stream << at(nRows_ - 1, nColumns_ - 1);
	}
	else {
		for (u32 row = 0; row < nRows_ - 1; row++) {
			for (u32 col = 0; col < nColumns_ - 1; col++) {
				stream << at(row, col) << " ";
			}
			stream << at(row, nColumns_ - 1) << Core::IOStream::endl;
		}
		for (u32 col = 0; col < nColumns_ - 1; col++) {
			stream << at(nRows_ - 1, col) << " ";
		}
		stream << at(nRows_ - 1, nColumns_ - 1);
	}
}

template<typename T>
void Matrix<T>::write(const std::string& filename, bool transpose, bool scientific) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::out);
		writeBinaryHeader(stream, transpose);
		writeBinary(stream, transpose);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::out);
		writeAsciiHeader(stream, transpose);
		writeAscii(stream, transpose, scientific);
		stream.close();
	}
	else {
		Core::AsciiStream stream(filename, std::ios::out);
		writeAsciiHeader(stream, transpose);
		writeAscii(stream, transpose, scientific);
		stream.close();
	}
}

template<typename T>
void Matrix<T>::readHeader(Core::IOStream& stream, bool transpose) {
	require(stream.is_open());
	u32 nRows, nColumns;
	stream >> nRows;
	stream >> nColumns;
	if (transpose)
		resize(nColumns, nRows);
	else
		resize(nRows, nColumns);
}

template<typename T>
void Matrix<T>::read(Core::IOStream& stream, bool transpose) {
	f32 x;
	if (transpose) {
		for (u32 col = 0; col < nColumns_; col++) {
			for (u32 row = 0; row < nRows_; row++) {
				stream >> x;
				at(row, col) = (T)x;
			}
		}
	}
	else {
		for (u32 row = 0; row < nRows_; row++) {
			for (u32 col = 0; col < nColumns_; col++) {
				stream >> x;
				at(row, col) = (T)x;
			}
		}
	}
}

template<typename T>
void Matrix<T>::read(const std::string& filename, bool transpose) {
	if (Core::Utils::isBinary(filename)) {
		Core::BinaryStream stream(filename, std::ios::in);
		readHeader(stream, transpose);
		read(stream, transpose);
		stream.close();
	}
	else if (Core::Utils::isGz(filename)) {
		Core::CompressedStream stream(filename, std::ios::in);
		readHeader(stream, transpose);
		read(stream, transpose);
		stream.close();;
	}
	else {
		Core::AsciiStream stream(filename, std::ios::in);
		readHeader(stream, transpose);
		read(stream, transpose);
		stream.close();
	}
}



} // namespace (Math)


#endif /* MATH_MATRIX_HH_ */
