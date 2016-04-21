#ifndef MATH_LAPACK_HH_
#define MATH_LAPACK_HH_

#include <mkl.h> // use Intel MKL for lapack routines

namespace Math {

/**
 * syevr (Eigenvalue decomposition)
 *
 * @return 0 if eigenvalue decomposition was successful, else error code
 *
 * @param n size of the quadratic (n x n)-matrix A
 * @param A pointer to the symmetric (n x n)-matrix in column major, entries are overwritten by syevr
 * @param il,iu indices in ascending order of smallest and largest eigenvalues to be returned
 * @param w returns selected eigenvalues in ascending order (must be of size n)
 * @param z return selected eigenvectors in first m columns (must be of size n x iu-il+1)
 * @param isuppz return first and last non-zero element of eigenvector i (must be of size 2*(iu-il+1)), referenced only if il=1, iu=n
 *
 */
template<typename T>
inline int syevr(const int n, T* A, const int il, const int iu, T* w, T* z, int* isuppz);

template<>
inline int syevr(const int n, float* A, const int il, const int iu, float* w, float* z, int* isuppz) {
	int m;
	return LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', n, A, n, 0, 0, il, iu, (float)0, &m, w, z, n, isuppz);
}
template<>
inline int syevr(const int n, double* A, const int il, const int iu, double* w, double* z, int* isuppz) {
	int m;
	return LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', n, A, n, 0, 0, il, iu, (float)0, &m, w, z, n, isuppz);
}

/**
 * gesvd (singular value decomposition)
 *
 * @return 0 if svd was successful, else error code
 *
 * @param m number of rows of matrix A
 * @param n number of columns of matrix A
 * @param A pointer to the matrix to be decomposed
 * @param s pointer to a vector for the singular values
 * @param u pointer to a matrix containing the singular vectors
 *
 */
template<typename T>
inline int gesvd(const int m, const int n, T* A, T* s, T* u);

template<>
inline int gesvd(const int m, const int n, float* A, float* s, float* u) {
	float* vt = 0;
	float* superb = new float[std::min(m,n) - 1];
	int result = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'S', 'N', m, n, A, m, s, u, m, vt, n, superb);
	delete superb;
	return result;
}
template<>
inline int gesvd(const int m, const int n, double* A, double* s, double* u) {
	double* vt = 0;
	double* superb = new double[std::min(m,n) - 1];
	int result = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', m, n, A, m, s, u, m, vt, n, superb);
	delete superb;
	return result;
}

} // namespace

#endif /* MATH_LAPACK_HH_ */
