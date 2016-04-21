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

#ifndef NN_TYPES_HH_
#define NN_TYPES_HH_

#include <Core/Types.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>

namespace Nn {

// neural network matrix/vector types
typedef Math::CudaMatrix<Float> Matrix;
typedef Math::CudaVector<Float> Vector;
// neural network label vector type
typedef Math::CudaVector<u32> LabelVector;

} // namespace

#endif /* NN_TYPES_HH_ */
