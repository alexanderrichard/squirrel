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
 * CudnnDataStructure.hh
 *
 *  Created on: Sep 1, 2016
 *      Author: ahsan
 */

#ifndef MATH_CUDNNDATASTRUCTURE_HH_
#define MATH_CUDNNDATASTRUCTURE_HH_

#include "CudnnWrapper.hh"
#include "CudaMatrix.hh"
namespace Math {

namespace cuDNN {

class CudnnDataStructure {
private:
	static bool isInitialized_;
public:
	static cudnnTensorFormat_t tensorFormat_;
	static cudnnHandle_t cudnnHandle_;

	static void initialize();
};

} //namespace cuDNN

}//namespace Math



#endif /* MATH_CUDNNDATASTRUCTURE_HH_ */
