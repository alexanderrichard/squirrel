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
