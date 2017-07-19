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

#ifndef MATH_FAST_VECTOR_OPERATIONS_HH_
#define MATH_FAST_VECTOR_OPERATIONS_HH_

#include <Core/OpenMPWrapper.hh>
#include <Math/MultithreadingHelper.hh>
#include <functional>
#include <cmath>


namespace Math {


// TODO add multithreading for all methods

/*
 *  y = exp(x) (componentwise)
 *
 */

template <typename T>
inline void vr_exp(int n, T *x, T *y){
	for (int i = 0; i < n; i++){
		y[i] = exp(x[i]);
	}
}


template <typename T>
inline void mt_vr_exp(int n, T *x, T *y, int nThreads){
#pragma omp parallel for
	for (int i = 0; i < n; i++){
		y[i] = exp(x[i]);
	}
}


// TODO add Intel MKL


/*
 *  y = log(x) (componentwise)
 */

template <typename T>
inline void vr_log(int n, T *x, T *y){
	for (int i = 0; i < n; i++){
		y[i] = log(x[i]);
	}
}


/*
 *  z = x**y (componentwise)
 */

template <typename T>
inline void vr_powx(int n, T *x, T y, T *z){
	for (int i = 0; i < n; i++){
		z[i] = pow(x[i], y);
	}
}


} // namespace math

#endif /* MATH_FAST_VECTOR_OPERATIONS_HH_ */
