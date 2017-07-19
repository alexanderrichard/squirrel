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

#include <Test/UnitTest.hh>
#include <Math/CudaMatrix.hh>

class TestCudaMatrix : public Test::Fixture
{
};

template<typename T>
void compare(const Math::CudaMatrix<T>& A, const Math::CudaMatrix<T>& B, const T& delta, bool logOutput = false)
{
	require_eq(A.nRows(), B.nRows());
	require_eq(A.nColumns(), B.nColumns());

	for(u32 j=0; j<A.nColumns(); j++)
	{
		for(u32 i=0; i<A.nRows(); i++)
		{
			if(logOutput) {
				Core::Log::os()<<A.at(i,j)<<"\t"<<B.at(i,j);//<<"\tImage:"<<j<<"\tIndex"<<i;
			}
			else {
				EXPECT_DOUBLE_EQ(A.at(i,j), B.at(i,j), delta);
			}
		}
	}
}

TEST_F(Test, TestCudaMatrix, swapWithMatrix)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(1,2);
	B.resize(3,1);
	A.initComputation();
	B.initComputation();
	A.setToZero();
	A.addConstantElementwise(1.0);
	B.setToZero();
	B.addConstantElementwise(2.0);
	A.swap(B);
	A.finishComputation();
	B.finishComputation();
	EXPECT_EQ(3u, A.nRows());
	EXPECT_EQ(1u, A.nColumns());
	EXPECT_EQ(1u, B.nRows());
	EXPECT_EQ(2u, B.nColumns());
	for (u32 i = 0; i < A.nRows(); i++) {
		for (u32 j = 0; j < A.nColumns(); j++) {
			EXPECT_EQ(A.at(i,j), 2.0);
		}
	}
	for (u32 i = 0; i < B.nRows(); i++) {
		for (u32 j = 0; j < B.nColumns(); j++) {
			EXPECT_EQ(B.at(i,j), 1.0);
		}
	}
}

TEST_F(Test, TestCudaMatrix, swapWithVector)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> v;
	A.resize(2,2);
	v.resize(3);
	A.initComputation();
	v.initComputation();
	A.setToZero();
	A.addConstantElementwise(1.0);
	v.setToZero();
	v.addConstantElementwise(2.0);
	A.swap(v);
	A.finishComputation();
	v.finishComputation();
	EXPECT_EQ(3u, A.nRows());
	EXPECT_EQ(1u, A.nColumns());
	EXPECT_EQ(4u, v.nRows());
	for (u32 i = 0; i < A.nRows(); i++) {
		EXPECT_EQ(2.0, A.at(i,0));
	}
	for (u32 i = 0; i < v.nRows(); i++) {
		EXPECT_EQ(1.0, v.at(i));
	}
}

TEST_F(Test, TestCudaMatrix, add)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	B.resize(2,3);
	A.copyStructure(B);
	A.at(0,0) = 1.0;
	A.at(0,1) = 2.0;
	A.at(0,2) = 3.0;
	A.at(1,0) = 4.0;
	A.at(1,1) = 5.0;
	A.at(1,2) = 6.0;
	B.at(0,0) = 2.0;
	B.at(0,1) = 4.0;
	B.at(0,2) = 6.0;
	B.at(1,0) = 8.0;
	B.at(1,1) = 10.0;
	B.at(1,2) = 12.0;
	A.initComputation();
	B.initComputation();
	A.add(B, 0.5);
	A.finishComputation();
	EXPECT_EQ(2.0, A.at(0,0));
	EXPECT_EQ(4.0, A.at(0,1));
	EXPECT_EQ(6.0, A.at(0,2));
	EXPECT_EQ(8.0, A.at(1,0));
	EXPECT_EQ(10.0, A.at(1,1));
	EXPECT_EQ(12.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	B2.resize(2,3);
	A2.copyStructure(B2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 2.0;
	A2.at(0,2) = 3.0;
	A2.at(1,0) = 4.0;
	A2.at(1,1) = 5.0;
	A2.at(1,2) = 6.0;
	B2.at(0,0) = 2.0;
	B2.at(0,1) = 4.0;
	B2.at(0,2) = 6.0;
	B2.at(1,0) = 8.0;
	B2.at(1,1) = 10.0;
	B2.at(1,2) = 12.0;
	A2.initComputation();
	B2.initComputation();
	A2.add(B2, 0.5f);
	A2.finishComputation();
	EXPECT_EQ((f32)2.0, A2.at(0,0));
	EXPECT_EQ((f32)4.0, A2.at(0,1));
	EXPECT_EQ((f32)6.0, A2.at(0,2));
	EXPECT_EQ((f32)8.0, A2.at(1,0));
	EXPECT_EQ((f32)10.0, A2.at(1,1));
	EXPECT_EQ((f32)12.0, A2.at(1,2));

	Math::CudaMatrix<f64> A3;
	Math::CudaMatrix<f32> B3;
	B3.resize(2,3);
	A3.copyStructure(B3);
	A3.resize(2,3);
	A3.at(0,0) = 1.0;
	A3.at(0,1) = 2.0;
	A3.at(0,2) = 3.0;
	A3.at(1,0) = 4.0;
	A3.at(1,1) = 5.0;
	A3.at(1,2) = 6.0;
	B3.at(0,0) = 2.0;
	B3.at(0,1) = 4.0;
	B3.at(0,2) = 6.0;
	B3.at(1,0) = 8.0;
	B3.at(1,1) = 10.0;
	B3.at(1,2) = 12.0;
	A3.initComputation();
	B3.initComputation();
	A3.add(B3, 0.5f);
	A3.finishComputation();
	EXPECT_EQ(2.0, A3.at(0,0));
	EXPECT_EQ(4.0, A3.at(0,1));
	EXPECT_EQ(6.0, A3.at(0,2));
	EXPECT_EQ(8.0, A3.at(1,0));
	EXPECT_EQ(10.0, A3.at(1,1));
	EXPECT_EQ(12.0, A3.at(1,2));

}

TEST_F(Test, TestCudaMatrix, addMatrixProduct)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	Math::CudaMatrix<f64> C;
	A.resize(2,3);
	B.resize(3,2);
	C.resize(2,2);
	A.at(0,0) = 1.0;
	A.at(0,1) = 2.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = 1.0;
	A.at(1,1) = 6.0;
	A.at(1,2) = 2.0;
	B.at(0,0) = 0.0;
	B.at(0,1) = 2.0;
	B.at(1,0) = 1.0;
	B.at(1,1) = 4.0;
	B.at(2,0) = 2.0;
	B.at(2,1) = 1.0;
	C.at(0,0) = 0.0;
	C.at(0,1) = 4.0;
	C.at(1,0) = 2.0;
	C.at(1,1) = 6.0;
	A.initComputation();
	B.initComputation();
	C.initComputation();
	C.addMatrixProduct(A,B,1,2,false,false);
	C.finishComputation();
	EXPECT_EQ(4.0, C.at(0,0));
	EXPECT_EQ(24.0, C.at(0,1));
	EXPECT_EQ(22.0, C.at(1,0));
	EXPECT_EQ(62.0, C.at(1,1));

	A.finishComputation();
	B.finishComputation();

	A.at(0,0) = 1.0;
	A.at(0,1) = 2.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = 1.0;
	A.at(1,1) = 6.0;
	A.at(1,2) = 2.0;
	B.at(0,0) = 0.0;
	B.at(0,1) = 2.0;
	B.at(1,0) = 1.0;
	B.at(1,1) = 4.0;
	B.at(2,0) = 2.0;
	B.at(2,1) = 1.0;
	C.at(0,0) = 0.0;
	C.at(0,1) = 4.0;
	C.at(1,0) = 2.0;
	C.at(1,1) = 6.0;
	A.initComputation();
	B.initComputation();
	C.initComputation();
	C.addMatrixProduct(A,B,0.5,3,false,false);
	C.finishComputation();
	EXPECT_EQ(6.0, C.at(0,0));
	EXPECT_EQ(32.0, C.at(0,1));
	EXPECT_EQ(31.0, C.at(1,0));
	EXPECT_EQ(87.0, C.at(1,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	Math::CudaMatrix<f32> C2;
	A2.resize(2,3);
	B2.resize(3,2);
	C2.resize(2,2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 2.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = 1.0;
	A2.at(1,1) = 6.0;
	A2.at(1,2) = 2.0;
	B2.at(0,0) = 0.0;
	B2.at(0,1) = 2.0;
	B2.at(1,0) = 1.0;
	B2.at(1,1) = 4.0;
	B2.at(2,0) = 2.0;
	B2.at(2,1) = 1.0;
	C2.at(0,0) = 0.0;
	C2.at(0,1) = 4.0;
	C2.at(1,0) = 2.0;
	C2.at(1,1) = 6.0;
	A2.initComputation();
	B2.initComputation();
	C2.initComputation();
	C2.addMatrixProduct(A2,B2,1,2,false,false);
	C2.finishComputation();
	EXPECT_EQ((f32)4.0, C2.at(0,0));
	EXPECT_EQ((f32)24.0, C2.at(0,1));
	EXPECT_EQ((f32)22.0, C2.at(1,0));
	EXPECT_EQ((f32)62.0, C2.at(1,1));

	A2.finishComputation();
	B2.finishComputation();

	A2.at(0,0) = 1.0;
	A2.at(0,1) = 2.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = 1.0;
	A2.at(1,1) = 6.0;
	A2.at(1,2) = 2.0;
	B2.at(0,0) = 0.0;
	B2.at(0,1) = 2.0;
	B2.at(1,0) = 1.0;
	B2.at(1,1) = 4.0;
	B2.at(2,0) = 2.0;
	B2.at(2,1) = 1.0;
	C2.at(0,0) = 0.0;
	C2.at(0,1) = 4.0;
	C2.at(1,0) = 2.0;
	C2.at(1,1) = 6.0;
	A2.initComputation();
	B2.initComputation();
	C2.initComputation();
	C2.addMatrixProduct(A2,B2,0.5,3,false,false);
	C2.finishComputation();
	EXPECT_EQ((f32)6.0, C2.at(0,0));
	EXPECT_EQ((f32)32.0, C2.at(0,1));
	EXPECT_EQ((f32)31.0, C2.at(1,0));
	EXPECT_EQ((f32)87.0, C2.at(1,1));
}

TEST_F(Test, TestCudaMatrix, sumOfSquares)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	A.initComputation();
	EXPECT_EQ(149.0, A.sumOfSquares());

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	A2.initComputation();
	EXPECT_EQ((f32)149.0, A2.sumOfSquares());
}

TEST_F(Test, TestCudaMatrix, dot)
{
	Math::CudaMatrix<f32> A;
	Math::CudaMatrix<f32> B;
	A.resize(2,3);
	B.resize(2,3);
	A.at(0,0) = 1.0;
	A.at(0,1) = 2.0;
	A.at(0,2) = 4.0;
	A.at(1,0) = 6.0;
	A.at(1,1) = 0.0;
	A.at(1,2) = 4.0;
	B.at(0,0) = 2.0;
	B.at(0,1) = 0.0;
	B.at(0,2) = 1.0;
	B.at(1,0) = 5.0;
	B.at(1,1) = 6.0;
	B.at(1,2) = 0.0;
	A.initComputation();
	B.initComputation();
	f64 result = A.dot(B);
	EXPECT_EQ(36.0, result);

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(2,3);
	B2.resize(2,3);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 2.0;
	A2.at(0,2) = 4.0;
	A2.at(1,0) = 6.0;
	A2.at(1,1) = 0.0;
	A2.at(1,2) = 4.0;
	B2.at(0,0) = 2.0;
	B2.at(0,1) = 0.0;
	B2.at(0,2) = 1.0;
	B2.at(1,0) = 5.0;
	B2.at(1,1) = 6.0;
	B2.at(1,2) = 0.0;
	A2.initComputation();
	B2.initComputation();
	result = A2.dot(B2);
	EXPECT_EQ(36.0, result);
}

TEST_F(Test, TestCudaMatrix, dotWithColumn)
{
	Math::CudaMatrix<f32> A;
	Math::CudaVector<f32> b;
	A.resize(2,3);
	b.resize(2);
	A.at(0,0) = 1.0;
	A.at(0,1) = 2.0;
	A.at(0,2) = 4.0;
	A.at(1,0) = 6.0;
	A.at(1,1) = 1.0;
	A.at(1,2) = 4.0;
	b.at(0) = 2.0;
	b.at(1) = 3.0;
	A.initComputation();
	b.initComputation();
	f32 result = A.dotWithColumn(b, 1);
	EXPECT_EQ(7.0f, result);

	Math::CudaMatrix<f64> A2;
	Math::CudaVector<f64> b2;
	A2.resize(2,3);
	b2.resize(2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 2.0;
	A2.at(0,2) = 4.0;
	A2.at(1,0) = 6.0;
	A2.at(1,1) = 1.0;
	A2.at(1,2) = 4.0;
	b2.at(0) = 2.0;
	b2.at(1) = 3.0;
	A2.initComputation();
	b2.initComputation();
	f64 result2 = A2.dotWithColumn(b2, 1);
	EXPECT_EQ(7.0, result2);

}

TEST_F(Test, TestCudaMatrix, scale)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(0,1) = 0.0;
	A.at(0,2) = 5.0;
	A.at(1,0) = 1.0;
	A.at(1,1) = 1.8;
	A.at(1,2) = 6.0;
	A.initComputation();
	A.scale(0.5);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(0.0, A.at(0,1));
	EXPECT_EQ(2.5, A.at(0,2));
	EXPECT_EQ(0.5, A.at(1,0));
	EXPECT_EQ(0.9, A.at(1,1));
	EXPECT_EQ(3.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(0,1) = 0.0;
	A2.at(0,2) = 5.0;
	A2.at(1,0) = 1.0;
	A2.at(1,1) = 1.8;
	A2.at(1,2) = 6.0;
	A2.initComputation();
	A2.scale(0.5);
	A2.finishComputation();
	EXPECT_EQ((f32)1.0, A2.at(0,0));
	EXPECT_EQ((f32)0.0, A2.at(0,1));
	EXPECT_EQ((f32)2.5, A2.at(0,2));
	EXPECT_EQ((f32)0.5, A2.at(1,0));
	EXPECT_EQ((f32)0.9, A2.at(1,1));
	EXPECT_EQ((f32)3.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, copy)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(2,3);
	B.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(0,1) = 8.0;
	A.at(0,2) = 4.0;
	A.at(1,0) = 10.0;
	A.at(1,1) = 6.0;
	A.at(1,2) = 12.0;
	B.initComputation();
	A.initComputation();
	B.copy(A);
	B.finishComputation();
	EXPECT_EQ(2.0, B.at(0,0));
	EXPECT_EQ(8.0, B.at(0,1));
	EXPECT_EQ(4.0, B.at(0,2));
	EXPECT_EQ(10.0, B.at(1,0));
	EXPECT_EQ(6.0, B.at(1,1));
	EXPECT_EQ(12.0, B.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(2,3);
	B2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(0,1) = 8.0;
	A2.at(0,2) = 4.0;
	A2.at(1,0) = 10.0;
	A2.at(1,1) = 6.0;
	A2.at(1,2) = 12.0;
	B2.initComputation();
	A2.initComputation();
	B2.copy(A2);
	B2.finishComputation();
	EXPECT_EQ((f32)2.0, B2.at(0,0));
	EXPECT_EQ((f32)8.0, B2.at(0,1));
	EXPECT_EQ((f32)4.0, B2.at(0,2));
	EXPECT_EQ((f32)10.0, B2.at(1,0));
	EXPECT_EQ((f32)6.0, B2.at(1,1));
	EXPECT_EQ((f32)12.0, B2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, copyArray)
{
	Math::CudaMatrix<f64> A;
	std::vector<f64> x(3);
	x.at(0) = 1;
	x.at(1) = 2;
	x.at(2) = 3;
	A.resize(2,3);
	A.initComputation();
	A.setToZero();
	A.copy(&(x.at(0)), 1, 1);
	A.finishComputation();
	EXPECT_EQ(0.0, A.at(0,0));
	EXPECT_EQ(0.0, A.at(0,1));
	EXPECT_EQ(2.0, A.at(0,2));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(1.0, A.at(1,1));
	EXPECT_EQ(3.0, A.at(1,2));

	//    Math::CudaMatrix<f32> A2;
	//    std::vector<f32> x2(3);
	//    x2.at(0) = 1;
	//    x2.at(1) = 2;
	//    x2.at(2) = 3;
	//    A2.resize(2,3);
	//    A2.initComputation();
	//    A2.setToZero();
	//    A2.copy(&(x2.at(0)), 1, 1);
	//    A2.finishComputation();
	//    EXPECT_EQ(0.0f, A2.at(0,0));
	//    EXPECT_EQ(0.0f, A2.at(0,1));
	//    EXPECT_EQ(2.0f, A2.at(0,2));
	//    EXPECT_EQ(0.0f, A2.at(1,0));
	//    EXPECT_EQ(1.0f, A2.at(1,1));
	//    EXPECT_EQ(3.0f, A2.at(1,2));
	//
}


//TEST_F(Test, TestCudaMatrix, multiply)
//{
//	Math::CudaMatrix<f64> A;
//	Math::CudaVector<f64> x;
//	Math::CudaVector<f64> y;
//	A.resize(2,3);
//	x.resize(3);
//	y.resize(2);
//	A.at(0,0) = 2.0;
//	A.at(1,0) = -2.0;
//	A.at(0,1) = 4.0;
//	A.at(1,1) = -8.0;
//	A.at(0,2) = 5.0;
//	A.at(1,2) = -6.0;
//	x.at(0) = 2.0;
//	x.at(1) = -2.0;
//	x.at(2) = -1.0;
//	y.at(0) = 1.0;
//	y.at(1) = 2.0;
//	A.initComputation();
//	x.initComputation();
//	y.initComputation();
//	A.multiply(x, y, false, 2, 2, 0);
//	y.finishComputation();
//	EXPECT_EQ(-16.0, y.at(0));
//	EXPECT_EQ(40.0, y.at(1));
//
//	Math::CudaMatrix<f32> A2;
//	Math::CudaVector<f32> x2;
//	Math::CudaVector<f32> y2;
//	A2.resize(2,3);
//	x2.resize(3);
//	y2.resize(2);
//	A2.at(0,0) = 2.0;
//	A2.at(1,0) = -2.0;
//	A2.at(0,1) = 4.0;
//	A2.at(1,1) = -8.0;
//	A2.at(0,2) = 5.0;
//	A2.at(1,2) = -6.0;
//	x2.at(0) = 2.0;
//	x2.at(1) = -2.0;
//	x2.at(2) = -1.0;
//	y2.at(0) = 1.0;
//	y2.at(1) = 2.0;
//	A2.initComputation();
//	x2.initComputation();
//	y2.initComputation();
//	A2.multiply(x2, y2, false, 2, 2, 0);
//	y2.finishComputation();
//	EXPECT_EQ((f32)-16.0, y2.at(0));
//	EXPECT_EQ((f32)40.0, y2.at(1));
//}

TEST_F(Test, TestCudaMatrix, l1norm)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	A.initComputation();
	EXPECT_EQ(27.0, A.l1norm());

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	A2.initComputation();
	EXPECT_EQ((f32)27.0, A2.l1norm());
}

TEST_F(Test, TestCudaMatrix, sum)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	A.initComputation();
	EXPECT_EQ(-5.0, A.sum());

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	A2.initComputation();
	EXPECT_EQ((f32)-5.0, A2.sum());
}

TEST_F(Test, TestCudaMatrix, argAbsMax)
{
	Math::CudaMatrix<f64> A;
	A.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(2,0) = 3.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = -3.0;
	A.at(2,1) = 2.0;
	A.initComputation();
	EXPECT_EQ((u32)2, A.argAbsMax(0));
	EXPECT_EQ((u32)1, A.argAbsMax(1));

	Math::CudaMatrix<f32> A2;
	A2.resize(3,2);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(2,0) = 3.0;
	A2.at(0,1) = 0.0;
	A2.at(1,1) = -3.0;
	A2.at(2,1) = 2.0;
	A2.initComputation();
	EXPECT_EQ((u32)2, A2.argAbsMax(0));
	EXPECT_EQ((u32)1, A2.argAbsMax(1));
}

TEST_F(Test, TestCudaMatrix, argAbsMin)
{
	Math::CudaMatrix<f64> A;
	A.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(2,0) = 3.0;
	A.at(0,1) = 1.0;
	A.at(1,1) = -3.0;
	A.at(2,1) = 0.0;
	A.initComputation();
	EXPECT_EQ((u32)0, A.argAbsMin(0));
	EXPECT_EQ((u32)2, A.argAbsMin(1));

	Math::CudaMatrix<f32> A2;
	A2.resize(3,2);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(2,0) = 3.0;
	A2.at(0,1) = 1.0;
	A2.at(1,1) = -3.0;
	A2.at(2,1) = 0.0;
	A2.initComputation();
	EXPECT_EQ((u32)0, A2.argAbsMin(0));
	EXPECT_EQ((u32)2, A2.argAbsMin(1));
}

TEST_F(Test, TestCudaMatrix, argMax)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<u32> v;
	A.resize(3,2);
	v.resize(2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -4.0;
	A.at(2,0) = 3.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	A.at(2,1) = -3.0;
	A.initComputation();
	v.initComputation();
	v.setToZero();
	A.argMax(v);
	v.finishComputation();
	EXPECT_EQ((u32)2, v.at(0));
	EXPECT_EQ((u32)1, v.at(1));

	Math::CudaMatrix<f32> A1;
	A1.resize(3,2);
	A1.at(0,0) = 2.0;
	A1.at(1,0) = -4.0;
	A1.at(2,0) = 3.0;
	A1.at(0,1) = 0.0;
	A1.at(1,1) = 2.0;
	A1.at(2,1) = -3.0;
	A1.initComputation();
	v.initComputation();
	v.setToZero();
	A1.argMax(v);
	v.finishComputation();
	EXPECT_EQ((u32)2, v.at(0));
	EXPECT_EQ((u32)1, v.at(1));
}

TEST_F(Test, TestCudaMatrix, maxValue)
{
	Math::CudaMatrix<f32> A;
	A.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -4.0;
	A.at(2,0) = 3.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	A.at(2,1) = -3.0;
	A.initComputation();
	f32 result = A.maxValue();
	EXPECT_EQ(3.0f, result);
}

TEST_F(Test, TestCudaMatrix, max)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -4.0;
	A.at(0,1) = -1.0;
	A.at(1,1) = -3.0;
	A.initComputation();
	A.max();
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(1.0, A.at(0,1));
	EXPECT_EQ(0.0, A.at(1,1));
}

TEST_F(Test, TestCudaMatrix, max2)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	Math::CudaMatrix<f64> C;
	A.resize(2,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -4.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	B.resize(2,2);
	B.at(0,0) = 3.0;
	B.at(1,0) = -5.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = 1.0;
	C.resize(2,2);
	A.initComputation();
	B.initComputation();
	C.initComputation();
	C.max(A, B);
	C.finishComputation();
	EXPECT_EQ(3.0, C.at(0,0));
	EXPECT_EQ(-4.0, C.at(1,0));
	EXPECT_EQ(0.0, C.at(0,1));
	EXPECT_EQ(2.0, C.at(1,1));
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplicationWithKroneckerDelta)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	Math::CudaMatrix<f64> C;
	A.resize(2,2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -4.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	B.resize(2,2);
	B.at(0,0) = 3.0;
	B.at(1,0) = -4.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = 2.0;
	C.resize(2,2);
	A.initComputation();
	B.initComputation();
	C.initComputation();
	C.fill(2.0);
	C.elementwiseMultiplicationWithKroneckerDelta(A, B);
	C.finishComputation();
	EXPECT_EQ(0.0, C.at(0,0));
	EXPECT_EQ(2.0, C.at(1,0));
	EXPECT_EQ(0.0, C.at(0,1));
	EXPECT_EQ(2.0, C.at(1,1));
}

TEST_F(Test, TestCudaMatrix, addOuterProduct)
{
	Math::CudaMatrix<f64> A(2,3);
	Math::CudaVector<f64> x(2);
	Math::CudaVector<f64> y(3);
	A.at(0,0) = 1.0;
	A.at(0,1) = 3.0;
	A.at(0,2) = 8.0;
	A.at(1,0) = 2.0;
	A.at(1,1) = 1.0;
	A.at(1,2) = 0.5;
	x.at(0) = 2.0;
	x.at(1) = 3.0;
	y.at(0) = 2.0;
	y.at(1) = 4.0;
	y.at(2) = 1.0;
	A.initComputation();
	x.initComputation();
	y.initComputation();
	A.addOuterProduct(x,y,0.5);
	A.finishComputation();
	EXPECT_EQ(3.0, A.at(0,0));
	EXPECT_EQ(7.0, A.at(0,1));
	EXPECT_EQ(9.0, A.at(0,2));
	EXPECT_EQ(5.0, A.at(1,0));
	EXPECT_EQ(7.0, A.at(1,1));
	EXPECT_EQ(2.0, A.at(1,2));

	Math::CudaMatrix<f64> B(3,2);
	Math::CudaVector<f64> z(3);
	Math::CudaVector<f64> t(2);
	B.at(0,0) = 8.0;
	B.at(0,1) = 2.0;
	B.at(1,0) = 5.0;
	B.at(1,1) = 1.0;
	B.at(2,0) = 4.0;
	B.at(2,1) = 1.5;
	z.at(0) = 1.0;
	z.at(1) = 0.0;
	z.at(2) = 2.0;
	t.at(0) = 3.0;
	t.at(1) = 8.0;
	B.initComputation();
	z.initComputation();
	t.initComputation();
	B.addOuterProduct(z,t,0.0);
	B.finishComputation();
	EXPECT_EQ(8.0, B.at(0,0));
	EXPECT_EQ(2.0, B.at(0,1));
	EXPECT_EQ(5.0, B.at(1,0));
	EXPECT_EQ(1.0, B.at(1,1));
	EXPECT_EQ(4.0, B.at(2,0));
	EXPECT_EQ(1.5, B.at(2,1));

	Math::CudaMatrix<f32> A2(2,3);
	Math::CudaVector<f32> x2(2);
	Math::CudaVector<f32> y2(3);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 3.0;
	A2.at(0,2) = 8.0;
	A2.at(1,0) = 2.0;
	A2.at(1,1) = 1.0;
	A2.at(1,2) = 0.5;
	x2.at(0) = 2.0;
	x2.at(1) = 3.0;
	y2.at(0) = 2.0;
	y2.at(1) = 4.0;
	y2.at(2) = 1.0;
	A2.initComputation();
	x2.initComputation();
	y2.initComputation();
	A2.addOuterProduct(x2,y2,0.5);
	A2.finishComputation();
	EXPECT_EQ((f32)3.0, A2.at(0,0));
	EXPECT_EQ((f32)7.0, A2.at(0,1));
	EXPECT_EQ((f32)9.0, A2.at(0,2));
	EXPECT_EQ((f32)5.0, A2.at(1,0));
	EXPECT_EQ((f32)7.0, A2.at(1,1));
	EXPECT_EQ((f32)2.0, A2.at(1,2));

	Math::CudaMatrix<f32> B2(3,2);
	Math::CudaVector<f32> z2(3);
	Math::CudaVector<f32> t2(2);
	B2.at(0,0) = 8.0;
	B2.at(0,1) = 2.0;
	B2.at(1,0) = 5.0;
	B2.at(1,1) = 1.0;
	B2.at(2,0) = 4.0;
	B2.at(2,1) = 1.5;
	z2.at(0) = 1.0;
	z2.at(1) = 0.0;
	z2.at(2) = 2.0;
	t2.at(0) = 3.0;
	t2.at(1) = 8.0;
	B2.initComputation();
	z2.initComputation();
	t2.initComputation();
	B2.addOuterProduct(z2,t2,0.0);
	B2.finishComputation();
	EXPECT_EQ((f32)8.0, B2.at(0,0));
	EXPECT_EQ((f32)2.0, B2.at(0,1));
	EXPECT_EQ((f32)5.0, B2.at(1,0));
	EXPECT_EQ((f32)1.0, B2.at(1,1));
	EXPECT_EQ((f32)4.0, B2.at(2,0));
	EXPECT_EQ((f32)1.5, B2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, getRow)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	A.at(0,0) = 6.0;
	A.at(0,1) = -2.0;
	A.at(0,2) = 1.0;
	A.at(1,0) = 3.0;
	A.at(1,1) = 2.0;
	A.at(1,2) = 0.0;
	A.initComputation();
	b.initComputation();
	A.getRow(1,b);
	b.finishComputation();
	EXPECT_EQ(3u, b.size());
	EXPECT_EQ(3.0, b.at(0));
	EXPECT_EQ(2.0, b.at(1));
	EXPECT_EQ(0.0, b.at(2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	A2.at(0,0) = 6.0;
	A2.at(0,1) = -2.0;
	A2.at(0,2) = 1.0;
	A2.at(1,0) = 3.0;
	A2.at(1,1) = 2.0;
	A2.at(1,2) = 0.0;
	A2.initComputation();
	b2.initComputation();
	A2.getRow(1,b2);
	b2.finishComputation();
	EXPECT_EQ(3u, b2.size());
	EXPECT_EQ((f32)3.0, b2.at(0));
	EXPECT_EQ((f32)2.0, b2.at(1));
	EXPECT_EQ((f32)0.0, b2.at(2));
}

TEST_F(Test, TestCudaMatrix, getColumn)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	A.at(0,0) = 0.0;
	A.at(0,1) = 4.0;
	A.at(0,2) = -1.0;
	A.at(1,0) = 6.0;
	A.at(1,1) = 1.0;
	A.at(1,2) = 6.0;
	A.initComputation();
	b.initComputation();
	A.getColumn(2,b);
	b.finishComputation();
	EXPECT_EQ(2u, b.size());
	EXPECT_EQ(-1.0, b.at(0));
	EXPECT_EQ(6.0, b.at(1));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	A2.at(0,0) = 0.0;
	A2.at(0,1) = 4.0;
	A2.at(0,2) = -1.0;
	A2.at(1,0) = 6.0;
	A2.at(1,1) = 1.0;
	A2.at(1,2) = 6.0;
	A2.initComputation();
	b2.initComputation();
	A2.getColumn(2,b2);
	b2.finishComputation();
	EXPECT_EQ(2u, b2.size());
	EXPECT_EQ((f32)-1.0, b2.at(0));
	EXPECT_EQ((f32)6.0, b2.at(1));

	Math::CudaMatrix<f32> A3;
	Math::CudaVector<f32> b3;
	A3.resize(3,2);
	A3.at(0,0) = 0.0;
	A3.at(0,1) = 4.0;
	A3.at(1,0) = 6.0;
	A3.at(1,1) = 1.0;
	A3.at(2,0) = -1.0;
	A3.at(2,1) = 5.0;
	A3.initComputation();
	b3.initComputation();
	A3.getColumn(1,b3);
	b3.finishComputation();
	EXPECT_EQ(3u, b3.size());
	EXPECT_EQ((f32)4.0, b3.at(0));
	EXPECT_EQ((f32)1.0, b3.at(1));
	EXPECT_EQ((f32)5.0, b3.at(2));
}

TEST_F(Test, TestCudaMatrix, setColumn)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	A.at(0,0) = 0.0;
	A.at(0,1) = 4.0;
	A.at(0,2) = -1.0;
	A.at(1,0) = 6.0;
	A.at(1,1) = 1.0;
	A.at(1,2) = 6.0;
	A.initComputation();
	b.resize(2);
	b.at(0) = 10.0;
	b.at(1) = 11.0;
	b.initComputation();
	A.setColumn(2, b);
	A.finishComputation();
	EXPECT_EQ(0.0, A.at(0,0));
	EXPECT_EQ(4.0, A.at(0,1));
	EXPECT_EQ(10.0, A.at(0,2));
	EXPECT_EQ(6.0, A.at(1,0));
	EXPECT_EQ(1.0, A.at(1,1));
	EXPECT_EQ(11.0, A.at(1,2));
}

TEST_F(Test, TestCudaMatrix, setRow)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	A.at(0,0) = 0.0;
	A.at(0,1) = 4.0;
	A.at(0,2) = -1.0;
	A.at(1,0) = 6.0;
	A.at(1,1) = 1.0;
	A.at(1,2) = 6.0;
	A.initComputation();
	b.resize(3);
	b.at(0) = 10.0;
	b.at(1) = 11.0;
	b.at(2) = 12.0;
	b.initComputation();
	A.setRow(0, b);
	A.finishComputation();
	EXPECT_EQ(10.0, A.at(0,0));
	EXPECT_EQ(11.0, A.at(0,1));
	EXPECT_EQ(12.0, A.at(0,2));
	EXPECT_EQ(6.0, A.at(1,0));
	EXPECT_EQ(1.0, A.at(1,1));
	EXPECT_EQ(6.0, A.at(1,2));
}

TEST_F(Test, TestCudaMatrix, copyBlockToMatrix)
{
	Math::Matrix<f64> A(4,3);
	Math::CudaMatrix<f64> B(3,4);
	f64 val = 0.0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 3; j++){
			val = val + 1.0;
			A.at(i,j) = val;
		}
	}
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 4; j++){
			B.at(i,j) = 0.0;
		}
	}
	B.copyBlockFromMatrix(A, 2, 0, 0, 1, 2, 3);
	EXPECT_EQ(7.0, B.at(0,1));
	EXPECT_EQ(8.0, B.at(0,2));
	EXPECT_EQ(9.0, B.at(0,3));
	EXPECT_EQ(10.0, B.at(1,1));
	EXPECT_EQ(11.0, B.at(1,2));
	EXPECT_EQ(12.0, B.at(1,3));
}

TEST_F(Test, TestCudaMatrix, copyBlockToMatrixCuda)
{
	Math::CudaMatrix<f64> A(4,3);
	Math::CudaMatrix<f64> B(3,4);
	f64 val = 0.0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 3; j++){
			val = val + 1.0;
			A.at(i,j) = val;
		}
	}
	A.initComputation();
	B.initComputation();
	B.setToZero();
	B.copyBlockFromMatrix(A, 2, 0, 0, 1, 2, 3);
	B.finishComputation();
	EXPECT_EQ(7.0, B.at(0,1));
	EXPECT_EQ(8.0, B.at(0,2));
	EXPECT_EQ(9.0, B.at(0,3));
	EXPECT_EQ(10.0, B.at(1,1));
	EXPECT_EQ(11.0, B.at(1,2));
	EXPECT_EQ(12.0, B.at(1,3));
}


TEST_F(Test, TestCudaMatrix, addBlockToMatrix)
{
	Math::CudaMatrix<f64> A(4,3);
	Math::CudaMatrix<f64> B(3,4);
	f64 val = 0.0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 3; j++){
			val = val + 1.0;
			A.at(i,j) = val;
		}
	}
	A.initComputation();
	B.initComputation();
	B.fill(1.0);
	B.addBlockFromMatrix(A, 2, 0, 0, 1, 2, 3, (f64)2.0);
	B.finishComputation();
	EXPECT_EQ(1.0, B.at(0,0));
	EXPECT_EQ(15.0, B.at(0,1));
	EXPECT_EQ(17.0, B.at(0,2));
	EXPECT_EQ(19.0, B.at(0,3));
	EXPECT_EQ(1.0, B.at(1,0));
	EXPECT_EQ(21.0, B.at(1,1));
	EXPECT_EQ(23.0, B.at(1,2));
	EXPECT_EQ(25.0, B.at(1,3));
	EXPECT_EQ(1.0, B.at(2,0));
	EXPECT_EQ(1.0, B.at(2,1));
	EXPECT_EQ(1.0, B.at(2,2));
	EXPECT_EQ(1.0, B.at(2,3));
}


TEST_F(Test, TestCudaMatrix, setToZero)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.initComputation();
	A.setToZero();
	A.finishComputation();
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 1.5;
	A.at(1,0) = 7.0;
	A.at(1,1) = 13.0;
	A.at(1,2) = 2.5;
	A.at(2,0) = 5.0;
	A.at(2,1) = -1.0;
	A.at(2,2) = 4.0;
	A.at(3,0) = 3.0;
	A.at(3,1) = -4.0;
	A.at(3,2) = 8.0;
	EXPECT_EQ(A.nRows(),4u);
	EXPECT_EQ(A.nColumns(),3u);
	EXPECT_EQ(A.size(),12u);
	A.initComputation();
	A.setToZero();
	A.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A.at(i,j), 0.0);

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.initComputation();
	A2.setToZero();
	A2.finishComputation();
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 1.5;
	A2.at(1,0) = 7.0;
	A2.at(1,1) = 13.0;
	A2.at(1,2) = 2.5;
	A2.at(2,0) = 5.0;
	A2.at(2,1) = -1.0;
	A2.at(2,2) = 4.0;
	A2.at(3,0) = 3.0;
	A2.at(3,1) = -4.0;
	A2.at(3,2) = 8.0;
	EXPECT_EQ(A2.nRows(),4u);
	EXPECT_EQ(A2.nColumns(),3u);
	EXPECT_EQ(A2.size(),12u);
	A2.initComputation();
	A2.setToZero();
	A2.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A2.at(i,j), (f32)0.0);

	Math::CudaMatrix<f64> A3;
	A3.resize(4,3);
	A3.setToZero();
	A3.initComputation();
	A3.finishComputation();
	A3.at(0,0) = 4.0;
	A3.at(0,1) = 6.0;
	A3.at(0,2) = 1.5;
	A3.at(1,0) = 7.0;
	A3.at(1,1) = 13.0;
	A3.at(1,2) = 2.5;
	A3.at(2,0) = 5.0;
	A3.at(2,1) = -1.0;
	A3.at(2,2) = 4.0;
	A3.at(3,0) = 3.0;
	A3.at(3,1) = -4.0;
	A3.at(3,2) = 8.0;
	EXPECT_EQ(A3.nRows(),4u);
	EXPECT_EQ(A3.nColumns(),3u);
	EXPECT_EQ(A3.size(),12u);
	A3.initComputation();
	A3.setToZero();
	A3.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A3.at(i,j), 0.0);

	Math::CudaMatrix<f32> A4;
	A4.resize(4,3);
	A4.setToZero();
	A4.initComputation();
	A4.finishComputation();
	A4.at(0,0) = 4.0;
	A4.at(0,1) = 6.0;
	A4.at(0,2) = 1.5;
	A4.at(1,0) = 7.0;
	A4.at(1,1) = 13.0;
	A4.at(1,2) = 2.5;
	A4.at(2,0) = 5.0;
	A4.at(2,1) = -1.0;
	A4.at(2,2) = 4.0;
	A4.at(3,0) = 3.0;
	A4.at(3,1) = -4.0;
	A4.at(3,2) = 8.0;
	EXPECT_EQ(A4.nRows(),4u);
	EXPECT_EQ(A4.nColumns(),3u);
	EXPECT_EQ(A4.size(),12u);
	A4.initComputation();
	A4.setToZero();
	A4.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A4.at(i,j), (f32)0.0);

}

TEST_F(Test, TestCudaMatrix, fill)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.initComputation();
	A.fill(0);
	A.finishComputation();
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 1.5;
	A.at(1,0) = 7.0;
	A.at(1,1) = 13.0;
	A.at(1,2) = 2.5;
	A.at(2,0) = 5.0;
	A.at(2,1) = -1.0;
	A.at(2,2) = 4.0;
	A.at(3,0) = 3.0;
	A.at(3,1) = -4.0;
	A.at(3,2) = 8.0;
	EXPECT_EQ(A.nRows(),4u);
	EXPECT_EQ(A.nColumns(),3u);
	EXPECT_EQ(A.size(),12u);
	A.initComputation();
	A.fill(10.0);
	A.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A.at(i,j), 10.0);

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.initComputation();
	A2.fill(0);
	A2.finishComputation();
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 1.5;
	A2.at(1,0) = 7.0;
	A2.at(1,1) = 13.0;
	A2.at(1,2) = 2.5;
	A2.at(2,0) = 5.0;
	A2.at(2,1) = -1.0;
	A2.at(2,2) = 4.0;
	A2.at(3,0) = 3.0;
	A2.at(3,1) = -4.0;
	A2.at(3,2) = 8.0;
	EXPECT_EQ(A2.nRows(),4u);
	EXPECT_EQ(A2.nColumns(),3u);
	EXPECT_EQ(A2.size(),12u);
	A2.initComputation();
	A2.fill(10.0);
	A2.finishComputation();
	for (u32 i = 0; i < 4; i++)
		for (u32 j = 0; j < 3; j++)
			EXPECT_EQ(A2.at(i,j), (f32)10.0);
}

TEST_F(Test, TestCudaMatrix, fill2)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 1.5;
	A.at(1,0) = 7.0;
	A.at(1,1) = 13.0;
	A.at(1,2) = 2.5;
	A.at(2,0) = 5.0;
	A.at(2,1) = -1.0;
	A.at(2,2) = 4.0;
	A.at(3,0) = 3.0;
	A.at(3,1) = -4.0;
	A.at(3,2) = 8.0;
	A.initComputation();
	A.fill(2, 0, 1, 2, 10.0);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 4.0);
	EXPECT_EQ(A.at(0,1), 10.0);
	EXPECT_EQ(A.at(0,2), 10.0);
	EXPECT_EQ(A.at(1,0), 7.0);
	EXPECT_EQ(A.at(1,1), 10.0);
	EXPECT_EQ(A.at(1,2), 10.0);
	EXPECT_EQ(A.at(2,0), 10.0);
	EXPECT_EQ(A.at(2,1), 10.0);
	EXPECT_EQ(A.at(2,2), 4.0);
	EXPECT_EQ(A.at(3,0), 10.0);
	EXPECT_EQ(A.at(3,1), 10.0);
	EXPECT_EQ(A.at(3,2), 8.0);

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 1.5;
	A2.at(1,0) = 7.0;
	A2.at(1,1) = 13.0;
	A2.at(1,2) = 2.5;
	A2.at(2,0) = 5.0;
	A2.at(2,1) = -1.0;
	A2.at(2,2) = 4.0;
	A2.at(3,0) = 3.0;
	A2.at(3,1) = -4.0;
	A2.at(3,2) = 8.0;
	A2.initComputation();
	A2.fill(2, 0, 1, 2, 10.0);
	A2.finishComputation();
	EXPECT_EQ(A2.at(0,0), 4.0f);
	EXPECT_EQ(A2.at(0,1), 10.0f);
	EXPECT_EQ(A2.at(0,2), 10.0f);
	EXPECT_EQ(A2.at(1,0), 7.0f);
	EXPECT_EQ(A2.at(1,1), 10.0f);
	EXPECT_EQ(A2.at(1,2), 10.0f);
	EXPECT_EQ(A2.at(2,0), 10.0f);
	EXPECT_EQ(A2.at(2,1), 10.0f);
	EXPECT_EQ(A2.at(2,2), 4.0f);
	EXPECT_EQ(A2.at(3,0), 10.0f);
	EXPECT_EQ(A2.at(3,1), 10.0f);
	EXPECT_EQ(A2.at(3,2), 8.0f);
}

TEST_F(Test, TestCudaMatrix, ensureMinimalValue)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 1.5;
	A.at(1,0) = 7.0;
	A.at(1,1) = 13.0;
	A.at(1,2) = 2.5;
	A.at(2,0) = 5.0;
	A.at(2,1) = -1.0;
	A.at(2,2) = 4.0;
	A.at(3,0) = 3.0;
	A.at(3,1) = -4.0;
	A.at(3,2) = 8.0;
	A.initComputation();
	A.ensureMinimalValue(2.0);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 4.0);
	EXPECT_EQ(A.at(0,1), 6.0);
	EXPECT_EQ(A.at(0,2), 2.0);
	EXPECT_EQ(A.at(1,0), 7.0);
	EXPECT_EQ(A.at(1,1), 13.0);
	EXPECT_EQ(A.at(1,2), 2.5);
	EXPECT_EQ(A.at(2,0), 5.0);
	EXPECT_EQ(A.at(2,1), 2.0);
	EXPECT_EQ(A.at(2,2), 4.0);
	EXPECT_EQ(A.at(3,0), 3.0);
	EXPECT_EQ(A.at(3,1), 2.0);
	EXPECT_EQ(A.at(3,2), 8.0);

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 1.5;
	A2.at(1,0) = 7.0;
	A2.at(1,1) = 13.0;
	A2.at(1,2) = 2.5;
	A2.at(2,0) = 5.0;
	A2.at(2,1) = -1.0;
	A2.at(2,2) = 4.0;
	A2.at(3,0) = 3.0;
	A2.at(3,1) = -4.0;
	A2.at(3,2) = 8.0;
	A2.initComputation();
	A2.ensureMinimalValue(2.0);
	A2.finishComputation();
	EXPECT_EQ(A2.at(0,0), 4.0f);
	EXPECT_EQ(A2.at(0,1), 6.0f);
	EXPECT_EQ(A2.at(0,2), 2.0f);
	EXPECT_EQ(A2.at(1,0), 7.0f);
	EXPECT_EQ(A2.at(1,1), 13.0f);
	EXPECT_EQ(A2.at(1,2), 2.5f);
	EXPECT_EQ(A2.at(2,0), 5.0f);
	EXPECT_EQ(A2.at(2,1), 2.0f);
	EXPECT_EQ(A2.at(2,2), 4.0f);
	EXPECT_EQ(A2.at(3,0), 3.0f);
	EXPECT_EQ(A2.at(3,1), 2.0f);
	EXPECT_EQ(A2.at(3,2), 8.0f);
}

TEST_F(Test, TestCudaMatrix, ensureMaximalValue)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 1.5;
	A.at(1,0) = 7.0;
	A.at(1,1) = 13.0;
	A.at(1,2) = 2.5;
	A.at(2,0) = 5.0;
	A.at(2,1) = -1.0;
	A.at(2,2) = 4.0;
	A.at(3,0) = 3.0;
	A.at(3,1) = -4.0;
	A.at(3,2) = 8.0;
	A.initComputation();
	A.ensureMaximalValue(2.0);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 2.0);
	EXPECT_EQ(A.at(0,1), 2.0);
	EXPECT_EQ(A.at(0,2), 1.5);
	EXPECT_EQ(A.at(1,0), 2.0);
	EXPECT_EQ(A.at(1,1), 2.0);
	EXPECT_EQ(A.at(1,2), 2.0);
	EXPECT_EQ(A.at(2,0), 2.0);
	EXPECT_EQ(A.at(2,1), -1.0);
	EXPECT_EQ(A.at(2,2), 2.0);
	EXPECT_EQ(A.at(3,0), 2.0);
	EXPECT_EQ(A.at(3,1), -4.0);
	EXPECT_EQ(A.at(3,2), 2.0);

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 1.5;
	A2.at(1,0) = 7.0;
	A2.at(1,1) = 13.0;
	A2.at(1,2) = 2.5;
	A2.at(2,0) = 5.0;
	A2.at(2,1) = -1.0;
	A2.at(2,2) = 4.0;
	A2.at(3,0) = 3.0;
	A2.at(3,1) = -4.0;
	A2.at(3,2) = 8.0;
	A2.initComputation();
	A2.ensureMaximalValue(2.0);
	A2.finishComputation();
	EXPECT_EQ(A2.at(0,0), 2.0f);
	EXPECT_EQ(A2.at(0,1), 2.0f);
	EXPECT_EQ(A2.at(0,2), 1.5f);
	EXPECT_EQ(A2.at(1,0), 2.0f);
	EXPECT_EQ(A2.at(1,1), 2.0f);
	EXPECT_EQ(A2.at(1,2), 2.0f);
	EXPECT_EQ(A2.at(2,0), 2.0f);
	EXPECT_EQ(A2.at(2,1), -1.0f);
	EXPECT_EQ(A2.at(2,2), 2.0f);
	EXPECT_EQ(A2.at(3,0), 2.0f);
	EXPECT_EQ(A2.at(3,1), -4.0f);
	EXPECT_EQ(A2.at(3,2), 2.0f);
}

TEST_F(Test, TestCudaMatrix, exp)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)i;
		A.at(1,i) = (double)-i;
	}
	A.initComputation();
	A.exp();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::exp((double)i), 0.0000000001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::exp((double)-i), 0.0000000001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (float)i;
		B.at(1,i) = (float)-i;
	}
	B.initComputation();
	B.exp();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(B.at(0,i), std::exp((float)i), 0.000001);
		EXPECT_DOUBLE_EQ(B.at(1,i), std::exp((float)-i), 0.000001);
	}
}

TEST_F(Test, TestCudaMatrix, signedPow)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,2);
	A.at(0,0) = 1.0;
	A.at(0,1) = -4.0;
	A.at(1,0) = 9.0;
	A.at(1,1) = -16.0;
	A.initComputation();
	A.signedPow(0.5);
	A.finishComputation();
	EXPECT_DOUBLE_EQ(1.0, A.at(0,0), 0.00000001);
	EXPECT_DOUBLE_EQ(-2.0, A.at(0,1), 0.00000001);
	EXPECT_DOUBLE_EQ(3.0, A.at(1,0), 0.00000001);
	EXPECT_DOUBLE_EQ(-4.0, A.at(1,1), 0.00000001);

	Math::CudaMatrix<f32> B;
	B.resize(2,2);
	B.at(0,0) = 1.0;
	B.at(0,1) = -4.0;
	B.at(1,0) = 9.0;
	B.at(1,1) = -16.0;
	B.initComputation();
	B.signedPow(0.5);
	B.finishComputation();
	EXPECT_DOUBLE_EQ(1.0, B.at(0,0), 0.00000001);
	EXPECT_DOUBLE_EQ(-2.0, B.at(0,1), 0.00000001);
	EXPECT_DOUBLE_EQ(3.0, B.at(1,0), 0.00000001);
	EXPECT_DOUBLE_EQ(-4.0, B.at(1,1), 0.00000001);
}

TEST_F(Test, TestCudaMatrix, log)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)(i + 1);
		A.at(1,i) = (double)2*(i + 1);
	}
	A.initComputation();
	A.log();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::log((double)(i + 1)), 0.0000000001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::log((double)2*(i + 1)), 0.0000000001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (float)(i + 1);
		B.at(1,i) = (float)2*(i + 1);
	}
	B.initComputation();
	B.log();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(B.at(0,i), std::log((float)(i + 1)), 0.000001);
		EXPECT_DOUBLE_EQ(B.at(1,i), std::log((float)2*(i + 1)), 0.000001);
	}
}

TEST_F(Test, TestCudaMatrix, sin)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)(i + 1);
		A.at(1,i) = (double)2*(i + 1);
	}
	A.initComputation();
	A.sin();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::sin((double)(i + 1)), 0.0000000001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::sin((double)2*(i + 1)), 0.0000000001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (float)(i + 1);
		B.at(1,i) = (float)2*(i + 1);
	}
	B.initComputation();
	B.sin();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(B.at(0,i), std::sin((float)(i + 1)), 0.000001);
		EXPECT_DOUBLE_EQ(B.at(1,i), std::sin((float)2*(i + 1)), 0.000001);
	}
}

TEST_F(Test, TestCudaMatrix, cos)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)(i + 1);
		A.at(1,i) = (double)2*(i + 1);
	}
	A.initComputation();
	A.cos();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::cos((double)(i + 1)), 0.0000000001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::cos((double)2*(i + 1)), 0.0000000001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (float)(i + 1);
		B.at(1,i) = (float)2*(i + 1);
	}
	B.initComputation();
	B.cos();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(B.at(0,i), std::cos((float)(i + 1)), 0.000001);
		EXPECT_DOUBLE_EQ(B.at(1,i), std::cos((float)2*(i + 1)), 0.000001);
	}
}

TEST_F(Test, TestCudaMatrix, asin)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)(i * 0.3);
		A.at(1,i) = (double)(-i * 0.3);
	}
	A.initComputation();
	A.asin();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::asin((double)(i * 0.3)), 0.00001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::asin((double)(-i * 0.3)), 0.00001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (double)(i * 0.3);
		B.at(1,i) = (double)(-i * 0.3);
	}
	B.initComputation();
	B.asin();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::asin((float)(i * 0.3)), 0.00001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::asin((float)(-i * 0.3)), 0.00001);
	}
}

TEST_F(Test, TestCudaMatrix, acos)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	for (int i = 0; i < 3; i++){
		A.at(0,i) = (double)(i * 0.3);
		A.at(1,i) = (double)(-i * 0.3);
	}
	A.initComputation();
	A.acos();
	A.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::acos((double)(i * 0.3)), 0.00001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::acos((double)(-i * 0.3)), 0.00001);
	}

	Math::CudaMatrix<f32> B;
	B.resize(2,3);
	for (int i = 0; i < 3; i++){
		B.at(0,i) = (double)(i * 0.3);
		B.at(1,i) = (double)(-i * 0.3);
	}
	B.initComputation();
	B.acos();
	B.finishComputation();
	for (int i = 0; i < 3; i++){
		EXPECT_DOUBLE_EQ(A.at(0,i), std::acos((float)(i * 0.3)), 0.00001);
		EXPECT_DOUBLE_EQ(A.at(1,i), std::acos((float)(-i * 0.3)), 0.00001);
	}
}

//TEST_F(Test, TestCudaMatrix, avgPool)
//{
//	Math::CudaMatrix<f64> A_64, RC_64, R_64;
//	Math::CudaMatrix<f32> A_32, RC_32, R_32;
//
//	A_64.read("./test-case-data/avgpool-forw-i.txt");
//	R_64.read("./test-case-data/avgpool-forw-r.txt");
//	A_32.read("./test-case-data/avgpool-forw-i.txt");
//	R_32.read("./test-case-data/avgpool-forw-r.txt");
//
//	RC_32.resize(R_32.nRows(), R_32.nColumns());
//	RC_64.resize(R_64.nRows(), R_64.nColumns());
//
//	A_64.initComputation();
//	A_32.initComputation();
//	RC_64.initComputation(false);
//	RC_32.initComputation(false);
//
//	RC_32.avgPool(A_32, 5, 5, 3, 2, 2);
//	RC_64.avgPool(A_64, 5, 5, 3, 2, 2);
//
//	A_64.finishComputation(false);
//	A_32.finishComputation(false);
//	RC_64.finishComputation();
//	RC_32.finishComputation();
//
//	Core::Log::openTag("avgpool-output");
//	compare(R_64, RC_64, 0.0000000001);
//	compare(R_32, RC_32, 0.0001f);
//	Core::Log::closeTag();
//}
//
//TEST_F(Test, TestCudaMatrix, backPropogateAvgPool)
//{
//	Math::CudaMatrix<f64> A_64, RC_64, R_64;
//	Math::CudaMatrix<f32> A_32, RC_32, R_32;
//
//	A_64.read("./test-case-data/avgpool-forw-r.txt");
//	A_32.read("./test-case-data/avgpool-forw-r.txt");
//	R_64.read("./test-case-data/avgpool-back-r.txt");
//	R_32.read("./test-case-data/avgpool-back-r.txt");
//
//	RC_32.resize(R_32.nRows(), R_32.nColumns());
//	RC_64.resize(R_64.nRows(), R_64.nColumns());
//
//	RC_32.setToZero();
//	RC_64.setToZero();
//
//	A_64.initComputation();
//	A_32.initComputation();
//	RC_64.initComputation();
//	RC_32.initComputation();
//
//	RC_32.backPropogateAvgPool(A_32, 5, 5, 3, 2, 2);
//	RC_64.backPropogateAvgPool(A_64, 5, 5, 3, 2, 2);
//
//	A_64.finishComputation(false);
//	A_32.finishComputation(false);
//	RC_64.finishComputation();
//	RC_32.finishComputation();
//
//	Core::Log::openTag("avgpool-backpropogate-output");
//	compare(R_64, RC_64, 0.0000000001);
//	compare(R_32, RC_32, 0.0001f);
//	Core::Log::closeTag();
//}

TEST_F(Test, TestCudaMatrix, maxPool)
{
	Math::CudaMatrix<f64> A_64, RC_64;
	Math::CudaMatrix<f32> A_32, R_32, RC_32;

	A_64.read("./test-case-data/maxpool-input.txt");
	A_32.read("./test-case-data/maxpool-input.txt");
	R_32.read("./test-case-data/maxpool-forward.txt");

	RC_64.resize(27, 10);
	RC_32.resize(27, 10);

	RC_64.setToZero();
	RC_32.setToZero();

	Core::Log::openTag("maxpool-output");

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation();
	RC_32.initComputation();

	RC_64.maxPool(A_64, 6, 5, 3, 2, 2);
	RC_32.maxPool(A_32, 6, 5, 3, 2, 2);

	A_64.finishComputation(false);
	A_32.finishComputation(false);
	RC_64.finishComputation();
	RC_32.finishComputation();

	for(u32 i=0; i<RC_32.nColumns(); i++)
	{
		for(u32 j=0; j<RC_32.nRows(); j++)
		{
			//Core::Log::os()<<RC_64.at(j,i)<<"\t"<<R_32.at(j,i);
			EXPECT_DOUBLE_EQ(R_32.at(j,i), RC_64.at(j,i),  0.0000000001);
			EXPECT_DOUBLE_EQ(R_32.at(j,i), RC_32.at(j,i), 0.00001);
		}
	}
	Core::Log::closeTag();
}

TEST_F(Test, TestCudaMatrix, backPropogateMaxPool)
{
	Math::CudaMatrix<f64> A_64, B_64, C_64, RC_64;
	Math::CudaMatrix<f32> A_32, B_32, C_32, RC_32, R;

	A_64.read("./test-case-data/maxpool-input.txt");
	A_32.read("./test-case-data/maxpool-input.txt");
	B_64.read("./test-case-data/maxpool-forward.txt");
	B_32.read("./test-case-data/maxpool-forward.txt");
	C_64.read("./test-case-data/maxpool-error-out.txt");
	C_32.read("./test-case-data/maxpool-error-out.txt");
	R.read("./test-case-data/maxpool-error-in.txt");

	RC_64.resize(90, 10);
	RC_32.resize(90, 10);

	RC_64.setToZero();
	RC_32.setToZero();

	A_64.initComputation();
	B_64.initComputation();
	C_64.initComputation();
	RC_64.initComputation();

	A_32.initComputation();
	B_32.initComputation();
	C_32.initComputation();
	RC_32.initComputation();

	RC_64.backPropogateMaxPool(A_64, B_64, C_64, 6, 5, 3, 2, 2);
	RC_32.backPropogateMaxPool(A_32, B_32, C_32, 6, 5, 3, 2, 2);

	A_64.finishComputation(false);
	B_64.finishComputation(false);
	C_64.finishComputation(false);
	RC_64.finishComputation();

	A_32.finishComputation(false);
	B_32.finishComputation(false);
	C_32.finishComputation(false);
	RC_32.finishComputation();

	//Core::Log::openTag("maxpool-backprop-output");

	for(u32 i=0; i<R.nColumns(); i++)
	{
		for(u32 j=0; j<R.nRows(); j++)
		{
			//Core::Log::os()<<R.at(j,i)<<"\t"<<RC_64.at(j,i)<<"\t"<<RC_32.at(j,i);
			EXPECT_DOUBLE_EQ(R.at(j,i), RC_64.at(j,i), 0.0000000001);
			EXPECT_DOUBLE_EQ(R.at(j,i), RC_32.at(j,i), 0.00001);
		}
	}
	//Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, reArange)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	R_64.read("./test-case-data/re-arange-forw-prop-r.txt");
	R_32.read("./test-case-data/re-arange-forw-prop-r.txt");
	A_64.read("./test-case-data/re-arange-back-prop-r.txt");
	A_32.read("./test-case-data/re-arange-back-prop-r.txt");

	RC_64.resize(R_64.nRows(), R_64.nColumns());
	RC_32.resize(R_64.nRows(), R_64.nColumns());

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_64.rearrange(A_64, 8);
	RC_32.rearrange(A_32, 8);

	A_64.finishComputation(false);
	A_32.finishComputation(false);
	RC_64.finishComputation();
	RC_32.finishComputation();

	Core::Log::openTag("re-arange-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);
	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, reArangeBackProp)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	A_64.read("./test-case-data/re-arange-forw-prop-r.txt");
	A_32.read("./test-case-data/re-arange-forw-prop-r.txt");
	R_64.read("./test-case-data/re-arange-back-prop-r.txt");
	R_32.read("./test-case-data/re-arange-back-prop-r.txt");

	RC_64.resize(R_64.nRows(), R_64.nColumns());
	RC_32.resize(R_64.nRows(), R_64.nColumns());

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_64.rearrangeBackProp(A_64, 3);
	RC_32.rearrangeBackProp(A_32, 3);

	A_64.finishComputation(false);
	A_32.finishComputation(false);
	RC_64.finishComputation();
	RC_32.finishComputation();

	Core::Log::openTag("re-arange-back-prop-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);
	Core::Log::closeTag();

}

TEST_F(Test, TestCudaMatrix, prepareConvolution)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	R_32.read("./test-case-data/prepare-conv-forw-r.txt");
	R_64.read("./test-case-data/prepare-conv-forw-r.txt");
	A_32.read("./test-case-data/prepare-conv-forw-i.txt");
	A_64.read("./test-case-data/prepare-conv-forw-i.txt");


	RC_64.resize(R_32.nRows(), R_32.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());

	A_64.initComputation();
	RC_64.initComputation(false);
	RC_64.prepareConvolution(A_64, 4, 5, 3, 3, 3,1,1);
	A_64.finishComputation(false);
	RC_64.finishComputation();

	A_32.initComputation();
	RC_32.initComputation(false);
	RC_32.prepareConvolution(A_32, 4, 5, 3, 3, 3,1,1);
	A_32.finishComputation(false);
	RC_32.finishComputation();

	Core::Log::openTag("prepareConvolution-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);
	Core::Log::closeTag();
}

TEST_F(Test, TestCudaMatrix, prepareConvolutionBackProp)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	R_64.read("./test-case-data/prepare-conv-back-r.txt");
	R_32.read("./test-case-data/prepare-conv-back-r.txt");
	A_64.read("./test-case-data/prepare-conv-forw-r.txt");
	A_32.read("./test-case-data/prepare-conv-forw-r.txt");

	RC_32.resize(R_32.nRows(), R_32.nColumns());
	RC_64.resize(R_32.nRows(), R_32.nColumns());

	//RC_32.setToZero();
	//RC_64.setToZero();

	Core::Log::openTag("prepareConvolution-backProp-output");

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_64.prepareConvolutionBackProp(A_64, 4, 5, 3, 3, 3);
	RC_32.prepareConvolutionBackProp(A_32, 4, 5, 3, 3, 3);

	RC_64.finishComputation();
	RC_32.finishComputation();
	A_64.finishComputation(false);
	A_32.finishComputation(false);

	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);

	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, prepareConvolutionStrided)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	A_64.read("./test-case-data/prepare-conv-strided-forw-i.txt");
	A_32.read("./test-case-data/prepare-conv-strided-forw-i.txt");
	R_64.read("./test-case-data/prepare-conv-strided-valid-forw-r.txt");
	R_32.read("./test-case-data/prepare-conv-strided-valid-forw-r.txt");

	RC_64.resize(R_64.nRows(), R_64.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_32.prepareConvolution(A_32, 6, 6, 3, 3, 3, 2, 2);
	RC_64.prepareConvolution(A_64, 6, 6, 3, 3, 3, 2, 2);

	RC_64.finishComputation();
	RC_32.finishComputation();
	A_64.finishComputation(false);
	A_32.finishComputation(false);

	Core::Log::openTag("prepareConvolution-strided-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);

	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, prepareConvolutionSameStridedBackProp)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	A_64.read("./test-case-data/prepare-conv-strided-same-forw-r.txt");
	A_32.read("./test-case-data/prepare-conv-strided-same-forw-r.txt");
	R_64.read("./test-case-data/prepare-conv-strided-same-back-r.txt");
	R_32.read("./test-case-data/prepare-conv-strided-same-back-r.txt");

	Core::Log::openTag("prepareConvolution-same-strided-output");

	RC_64.resize(R_64.nRows(), R_64.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_64.prepareConvolutionSameBackProp(A_64, 6, 6, 3, 3, 3, 2, 2);
	RC_32.prepareConvolutionSameBackProp(A_32, 6, 6, 3, 3, 3, 2, 2);

	RC_64.finishComputation();
	RC_32.finishComputation();
	A_64.finishComputation(false);
	A_32.finishComputation(false);

	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);

	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, prepareConvolutionSameStrided)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	A_64.read("./test-case-data/prepare-conv-strided-forw-i.txt");
	A_32.read("./test-case-data/prepare-conv-strided-forw-i.txt");
	R_64.read("./test-case-data/prepare-conv-strided-same-forw-r.txt");
	R_32.read("./test-case-data/prepare-conv-strided-same-forw-r.txt");

	Core::Log::openTag("prepareConvolution-same-strided-output");

	RC_64.resize(R_64.nRows(), R_64.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());

	A_64.initComputation();
	A_32.initComputation();
	RC_64.initComputation(false);
	RC_32.initComputation(false);

	RC_32.prepareConvolutionSame(A_32, 6, 6, 3, 3, 3, 2, 2);
	RC_64.prepareConvolutionSame(A_64, 6, 6, 3, 3, 3, 2, 2);

	RC_64.finishComputation();
	RC_32.finishComputation();
	A_64.finishComputation(false);
	A_32.finishComputation(false);

	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);

	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, prepareConvolutionSame)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	R_32.read("./test-case-data/prepare-conv-same-forw-r.txt");
	R_64.read("./test-case-data/prepare-conv-same-forw-r.txt");
	A_32.read("./test-case-data/prepare-conv-same-forw-i.txt");
	A_64.read("./test-case-data/prepare-conv-same-forw-i.txt");

	RC_64.resize(R_32.nRows(), R_32.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());

	A_64.initComputation();
	RC_64.initComputation(false);
	RC_64.prepareConvolutionSame(A_64, 3, 4, 3, 3, 3, 1, 1);
	A_64.finishComputation(false);
	RC_64.finishComputation();

	A_32.initComputation();
	RC_32.initComputation(false);
	RC_32.prepareConvolutionSame(A_32, 3, 4, 3, 3, 3, 1, 1);
	A_32.finishComputation(false);
	RC_32.finishComputation();

	Core::Log::openTag("prepareConvolution-same-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);
	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, prepareConvolutionSameBackProp)
{
	Math::CudaMatrix<f64> A_64, RC_64, R_64;
	Math::CudaMatrix<f32> A_32, RC_32, R_32;

	R_32.read("./test-case-data/prepare-conv-same-back-r.txt");
	R_64.read("./test-case-data/prepare-conv-same-back-r.txt");
	A_32.read("./test-case-data/prepare-conv-same-forw-r.txt");
	A_64.read("./test-case-data/prepare-conv-same-forw-r.txt");

	RC_64.resize(R_32.nRows(), R_32.nColumns());
	RC_32.resize(R_32.nRows(), R_32.nColumns());


	A_64.initComputation();
	RC_64.initComputation(false);
	RC_64.prepareConvolutionSameBackProp(A_64, 3, 4, 3, 3, 3, 1, 1);
	A_64.finishComputation(false);
	RC_64.finishComputation();

	A_32.initComputation();
	RC_32.initComputation(false);
	RC_32.prepareConvolutionSameBackProp(A_32, 3, 4, 3, 3, 3, 1, 1);
	A_32.finishComputation(false);
	RC_32.finishComputation();

	Core::Log::openTag("prepareConvolution-same-backprop-output");
	compare(R_64, RC_64, 0.0000000001);
	compare(R_32, RC_32, 0.0001f);
	Core::Log::closeTag();
}
TEST_F(Test, TestCudaMatrix, abs)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,2);
	A.at(0,0) = 1.0;
	A.at(0,1) = -2.0;
	A.at(1,0) = 3.0;
	A.at(1,1) = -4.0;
	A.initComputation();
	A.abs();
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(2.0, A.at(0,1));
	EXPECT_EQ(3.0, A.at(1,0));
	EXPECT_EQ(4.0, A.at(1,1));
}


TEST_F(Test, TestCudaMatrix, sigmoid)
{
	Math::CudaMatrix<f64> A;
	A.resize(3,2);
	A.at(0,0) = 1.0; A.at(0,1) = 2.0;
	A.at(1,0) = -1.0; A.at(1,1) = 3.0;
	A.at(2,0) = 4.0; A.at(2,1) = -4.0;

	A.initComputation();
	A.sigmoid();
	A.finishComputation();

	EXPECT_DOUBLE_EQ(0.7310585786300049, A.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.88079707797788231, A.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.2689414213699951, A.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.95257412682243336, A.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.98201379003790845, A.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.017986209962091559, A.at(2,1), 0.000001);

	A.at(0,0) = 0.5; A.at(0,1) = 1.0;
	A.at(1,0) = -0.5; A.at(1,1) = 1.5;
	A.at(2,0) = 2.0; A.at(2,1) = -2.0;

	A.initComputation();
	A.sigmoid(2.0);
	A.finishComputation();

	EXPECT_DOUBLE_EQ(0.7310585786300049, A.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.88079707797788231, A.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.2689414213699951, A.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.95257412682243336, A.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.98201379003790845, A.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.017986209962091559, A.at(2,1), 0.000001);

	// test behavior for very large values
	A.resize(1,3);
	A.at(0,0) = 710.0;
	A.at(0,1) = 0.0;
	A.at(0,2) = -710.0;

	A.initComputation();
	A.sigmoid();
	A.finishComputation();

	EXPECT_DOUBLE_EQ(1.0, A.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.5, A.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.0, A.at(0,2), 0.000001);


	Math::CudaMatrix<f32> A2;
	A2.resize(3,2);
	A2.at(0,0) = 1.0; A2.at(0,1) = 2.0;
	A2.at(1,0) = -1.0; A2.at(1,1) = 3.0;
	A2.at(2,0) = 4.0; A2.at(2,1) = -4.0;

	A2.initComputation();
	A2.sigmoid();
	A2.finishComputation();

	EXPECT_DOUBLE_EQ(0.7310585786300049, A2.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.88079707797788231, A2.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.2689414213699951, A2.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.95257412682243336, A2.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.98201379003790845, A2.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.017986209962091559, A2.at(2,1), 0.000001);

	A2.at(0,0) = 0.5; A2.at(0,1) = 1.0;
	A2.at(1,0) = -0.5; A2.at(1,1) = 1.5;
	A2.at(2,0) = 2.0; A2.at(2,1) = -2.0;

	A2.initComputation();
	A2.sigmoid(2.0);
	A2.finishComputation();

	EXPECT_DOUBLE_EQ(0.7310585786300049, A2.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.88079707797788231, A2.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.2689414213699951, A2.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.95257412682243336, A2.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.98201379003790845, A2.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.017986209962091559, A2.at(2,1), 0.000001);

	// test behavior for very large values
	A2.resize(1,3);
	A2.at(0,0) = 710.0;
	A2.at(0,1) = 0.0;
	A2.at(0,2) = -710.0;

	A2.initComputation();
	A2.sigmoid();
	A2.finishComputation();

	EXPECT_DOUBLE_EQ((f32)1.0, A2.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ((f32)0.5, A2.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ((f32)0.0, A2.at(0,2), 0.000001);

}

TEST_F(Test, TestCudaMatrix, triangle)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 3.0;
	A.at(0,1) = -1.0;
	A.at(0,2) = 0.4;
	A.at(1,0) = -0.2;
	A.at(1,1) = 0.5;
	A.at(1,2) = 0.0;
	A.initComputation();
	A.triangle();
	A.finishComputation();
	EXPECT_EQ(0.0, A.at(0,0));
	EXPECT_EQ(0.0, A.at(0,1));
	EXPECT_EQ(0.6, A.at(0,2));
	EXPECT_EQ(0.8, A.at(1,0));
	EXPECT_EQ(0.5, A.at(1,1));
	EXPECT_EQ(1.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 3.0;
	A2.at(0,1) = -1.0;
	A2.at(0,2) = 0.4;
	A2.at(1,0) = -0.2;
	A2.at(1,1) = 0.5;
	A2.at(1,2) = 0.0;
	A2.initComputation();
	A2.triangle();
	A2.finishComputation();
	EXPECT_EQ((f32)0.0, A2.at(0,0));
	EXPECT_EQ((f32)0.0, A2.at(0,1));
	EXPECT_EQ((f32)0.6, A2.at(0,2));
	EXPECT_EQ((f32)0.8, A2.at(1,0));
	EXPECT_EQ((f32)0.5, A2.at(1,1));
	EXPECT_EQ((f32)1.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, tanh)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 3.0;
	A.at(0,1) = -1.0;
	A.at(0,2) = 4.5;
	A.at(1,0) = 2.0;
	A.at(1,1) = 0.5;
	A.at(1,2) = 8.0;
	EXPECT_EQ(A.nRows(),2u);
	EXPECT_EQ(A.nColumns(),3u);
	EXPECT_EQ(A.size(),6u);
	A.initComputation();
	A.tanh();
	A.finishComputation();
	EXPECT_DOUBLE_EQ(0.995054753687, A.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.761594155955, A.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.999753210848, A.at(0,2), 0.000001);
	EXPECT_DOUBLE_EQ(0.964027580076, A.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.462117157266, A.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.99999977493, A.at(1,2), 0.000001);

	Math::CudaMatrix<f32> A2;
	A2.resize(2,3);
	A2.at(0,0) = 3.0;
	A2.at(0,1) = -1.0;
	A2.at(0,2) = 4.5;
	A2.at(1,0) = 2.0;
	A2.at(1,1) = 0.5;
	A2.at(1,2) = 8.0;
	EXPECT_EQ(A2.nRows(),2u);
	EXPECT_EQ(A2.nColumns(),3u);
	EXPECT_EQ(A2.size(),6u);
	A2.initComputation();
	A2.tanh();
	A2.finishComputation();
	EXPECT_DOUBLE_EQ(0.995054753687, A2.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.761594155955, A2.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.999753210848, A2.at(0,2), 0.000001);
	EXPECT_DOUBLE_EQ(0.964027580076, A2.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.462117157266, A2.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.99999977493, A2.at(1,2), 0.000001);
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplication)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(2,3);
	B.resize(2,3);
	A.at(0,0) = 1.0;
	A.at(0,1) = 5.0;
	A.at(0,2) = 2.0;
	A.at(1,0) = -1.0;
	A.at(1,1) = 0.5;
	A.at(1,2) = 2.0;
	B.at(0,0) = 7.0;
	B.at(0,1) = 12.0;
	B.at(0,2) = 0.5;
	B.at(1,0) = 3.0;
	B.at(1,1) = -2.0;
	B.at(1,2) = 0.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseMultiplication(B);
	A.finishComputation();
	EXPECT_EQ(7.0, A.at(0,0));
	EXPECT_EQ(60.0, A.at(0,1));
	EXPECT_EQ(1.0, A.at(0,2));
	EXPECT_EQ(-3.0, A.at(1,0));
	EXPECT_EQ(-1.0, A.at(1,1));
	EXPECT_EQ(0.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(2,3);
	B2.resize(2,3);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 5.0;
	A2.at(0,2) = 2.0;
	A2.at(1,0) = -1.0;
	A2.at(1,1) = 0.5;
	A2.at(1,2) = 2.0;
	B2.at(0,0) = 7.0;
	B2.at(0,1) = 12.0;
	B2.at(0,2) = 0.5;
	B2.at(1,0) = 3.0;
	B2.at(1,1) = -2.0;
	B2.at(1,2) = 0.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseMultiplication(B2);
	A2.finishComputation();
	EXPECT_EQ((f32)7.0, A2.at(0,0));
	EXPECT_EQ((f32)60.0, A2.at(0,1));
	EXPECT_EQ((f32)1.0, A2.at(0,2));
	EXPECT_EQ((f32)-3.0, A2.at(1,0));
	EXPECT_EQ((f32)-1.0, A2.at(1,1));
	EXPECT_EQ((f32)0.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, elementwiseDivision)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(2,3);
	B.resize(2,3);
	A.at(0,0) = 4.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 5.0;
	A.at(1,0) = 2.0;
	A.at(1,1) = 9.0;
	A.at(1,2) = 12.0;
	B.at(0,0) = 1.0;
	B.at(0,1) = 3.0;
	B.at(0,2) = 1.0;
	B.at(1,0) = 2.0;
	B.at(1,1) = 2.0;
	B.at(1,2) = 4.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseDivision(B);
	A.finishComputation();
	EXPECT_EQ(4.0, A.at(0,0));
	EXPECT_EQ(2.0, A.at(0,1));
	EXPECT_EQ(5.0, A.at(0,2));
	EXPECT_EQ(1.0, A.at(1,0));
	EXPECT_EQ(4.5, A.at(1,1));
	EXPECT_EQ(3.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(2,3);
	B2.resize(2,3);
	A2.at(0,0) = 4.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 5.0;
	A2.at(1,0) = 2.0;
	A2.at(1,1) = 9.0;
	A2.at(1,2) = 12.0;
	B2.at(0,0) = 1.0;
	B2.at(0,1) = 3.0;
	B2.at(0,2) = 1.0;
	B2.at(1,0) = 2.0;
	B2.at(1,1) = 2.0;
	B2.at(1,2) = 4.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseDivision(B2);
	A2.finishComputation();
	EXPECT_EQ((f32)4.0, A2.at(0,0));
	EXPECT_EQ((f32)2.0, A2.at(0,1));
	EXPECT_EQ((f32)5.0, A2.at(0,2));
	EXPECT_EQ((f32)1.0, A2.at(1,0));
	EXPECT_EQ((f32)4.5, A2.at(1,1));
	EXPECT_EQ((f32)3.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, addConstantElementwise)
{
	Math::CudaMatrix<f64> A;
	A.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(0,1) = 1.0;
	A.at(1,0) = 5.5;
	A.at(1,1) = -1.0;
	A.at(2,0) = 3.0;
	A.at(2,1) = 0.5;
	A.initComputation();
	A.addConstantElementwise(2.5);
	A.finishComputation();
	EXPECT_EQ(4.5, A.at(0,0));
	EXPECT_EQ(3.5, A.at(0,1));
	EXPECT_EQ(8.0, A.at(1,0));
	EXPECT_EQ(1.5, A.at(1,1));
	EXPECT_EQ(5.5, A.at(2,0));
	EXPECT_EQ(3.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	A2.resize(3,2);
	A2.at(0,0) = 2.0;
	A2.at(0,1) = 1.0;
	A2.at(1,0) = 5.5;
	A2.at(1,1) = -1.0;
	A2.at(2,0) = 3.0;
	A2.at(2,1) = 0.5;
	A2.initComputation();
	A2.addConstantElementwise(2.5);
	A2.finishComputation();
	EXPECT_EQ((f32)4.5, A2.at(0,0));
	EXPECT_EQ((f32)3.5, A2.at(0,1));
	EXPECT_EQ((f32)8.0, A2.at(1,0));
	EXPECT_EQ((f32)1.5, A2.at(1,1));
	EXPECT_EQ((f32)5.5, A2.at(2,0));
	EXPECT_EQ((f32)3.0, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplicationWithSigmoidDerivative)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(3,2);
	B.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(0,1) = 0.0;
	A.at(1,0) = 1.0;
	A.at(1,1) = 4.0;
	A.at(2,0) = -1.5;
	A.at(2,1) = 1.0;
	B.at(0,0) = 2.0;
	B.at(0,1) = 9.0;
	B.at(1,0) = -1.0;
	B.at(1,1) = 0.5;
	B.at(2,0) = 3.5;
	B.at(2,1) = 4.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseMultiplicationWithSigmoidDerivative(B);
	A.finishComputation();
	EXPECT_EQ(-4.0, A.at(0,0));
	EXPECT_EQ(0.0, A.at(0,1));
	EXPECT_EQ(-2.0, A.at(1,0));
	EXPECT_EQ(1.0, A.at(1,1));
	EXPECT_EQ(13.125, A.at(2,0));
	EXPECT_EQ(-12.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(3,2);
	B2.resize(3,2);
	A2.at(0,0) = 2.0;
	A2.at(0,1) = 0.0;
	A2.at(1,0) = 1.0;
	A2.at(1,1) = 4.0;
	A2.at(2,0) = -1.5;
	A2.at(2,1) = 1.0;
	B2.at(0,0) = 2.0;
	B2.at(0,1) = 9.0;
	B2.at(1,0) = -1.0;
	B2.at(1,1) = 0.5;
	B2.at(2,0) = 3.5;
	B2.at(2,1) = 4.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseMultiplicationWithSigmoidDerivative(B2);
	A2.finishComputation();
	EXPECT_EQ((f32)-4.0, A2.at(0,0));
	EXPECT_EQ((f32)0.0, A2.at(0,1));
	EXPECT_EQ((f32)-2.0, A2.at(1,0));
	EXPECT_EQ((f32)1.0, A2.at(1,1));
	EXPECT_EQ((f32)13.125, A2.at(2,0));
	EXPECT_EQ((f32)-12.0, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplicationWithTriangleDerivative)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(3,2);
	B.resize(3,2);
	A.at(0,0) = 2.0;
	A.at(0,1) = 1.0;
	A.at(1,0) = 2.0;
	A.at(1,1) = 4.0;
	A.at(2,0) = -1.5;
	A.at(2,1) = 1.0;
	B.at(0,0) = 2.0;
	B.at(0,1) = 0.9;
	B.at(1,0) = -0.1;
	B.at(1,1) = 0.5;
	B.at(2,0) = 3.5;
	B.at(2,1) = 4.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseMultiplicationWithTriangleDerivative(B);
	A.finishComputation();
	EXPECT_EQ(0.0, A.at(0,0));
	EXPECT_EQ(-1.0, A.at(0,1));
	EXPECT_EQ(2.0, A.at(1,0));
	EXPECT_EQ(-4.0, A.at(1,1));
	EXPECT_EQ(0.0, A.at(2,0));
	EXPECT_EQ(0.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(3,2);
	B2.resize(3,2);
	A2.at(0,0) = 2.0;
	A2.at(0,1) = 1.0;
	A2.at(1,0) = 2.0;
	A2.at(1,1) = 4.0;
	A2.at(2,0) = -1.5;
	A2.at(2,1) = 1.0;
	B2.at(0,0) = 2.0;
	B2.at(0,1) = 0.9;
	B2.at(1,0) = -0.1;
	B2.at(1,1) = 0.5;
	B2.at(2,0) = 3.5;
	B2.at(2,1) = 4.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseMultiplicationWithTriangleDerivative(B2);
	A2.finishComputation();
	EXPECT_EQ((f32)0.0, A2.at(0,0));
	EXPECT_EQ((f32)-1.0, A2.at(0,1));
	EXPECT_EQ((f32)2.0, A2.at(1,0));
	EXPECT_EQ((f32)-4.0, A2.at(1,1));
	EXPECT_EQ((f32)0.0, A2.at(2,0));
	EXPECT_EQ((f32)0.0, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplicationWithTanhDerivative)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(3,2);
	B.resize(3,2);
	A.at(0,0) = 1.0;
	A.at(0,1) = 3.0;
	A.at(1,0) = 0.0;
	A.at(1,1) = 2.0;
	A.at(2,0) = -1.5;
	A.at(2,1) = 3.0;
	B.at(0,0) = 0.5;
	B.at(0,1) = 2.0;
	B.at(1,0) = -1.0;
	B.at(1,1) = 2.5;
	B.at(2,0) = 2.0;
	B.at(2,1) = 5.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseMultiplicationWithTanhDerivative(B);
	A.finishComputation();
	EXPECT_EQ(0.75, A.at(0,0));
	EXPECT_EQ(-9.0, A.at(0,1));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(-10.5, A.at(1,1));
	EXPECT_EQ(4.5, A.at(2,0));
	EXPECT_EQ(-72.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(3,2);
	B2.resize(3,2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 3.0;
	A2.at(1,0) = 0.0;
	A2.at(1,1) = 2.0;
	A2.at(2,0) = -1.5;
	A2.at(2,1) = 3.0;
	B2.at(0,0) = 0.5;
	B2.at(0,1) = 2.0;
	B2.at(1,0) = -1.0;
	B2.at(1,1) = 2.5;
	B2.at(2,0) = 2.0;
	B2.at(2,1) = 5.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseMultiplicationWithTanhDerivative(B2);
	A2.finishComputation();
	EXPECT_EQ((f32)0.75, A2.at(0,0));
	EXPECT_EQ((f32)-9.0, A2.at(0,1));
	EXPECT_EQ((f32)0.0, A2.at(1,0));
	EXPECT_EQ((f32)-10.5, A2.at(1,1));
	EXPECT_EQ((f32)4.5, A2.at(2,0));
	EXPECT_EQ((f32)-72.0, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, elementwiseMultiplicationWithClippedDerivative)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(3,2);
	B.resize(3,2);
	A.at(0,0) = 1.0;
	A.at(0,1) = -3.0;
	A.at(1,0) = 0.0;
	A.at(1,1) = 2.0;
	A.at(2,0) = -1.5;
	A.at(2,1) = 3.0;
	B.at(0,0) = 0.5;
	B.at(0,1) = 2.0;
	B.at(1,0) = -1.0;
	B.at(1,1) = 2.5;
	B.at(2,0) = -2.0;
	B.at(2,1) = 5.0;
	A.initComputation();
	B.initComputation();
	A.elementwiseMultiplicationWithClippedDerivative(B, -1.5, 3.0);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(-3.0, A.at(0,1));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(2.0, A.at(1,1));
	EXPECT_EQ(0.0, A.at(2,0));
	EXPECT_EQ(0.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(3,2);
	B2.resize(3,2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = -3.0;
	A2.at(1,0) = 0.0;
	A2.at(1,1) = 2.0;
	A2.at(2,0) = -1.5;
	A2.at(2,1) = 3.0;
	B2.at(0,0) = 0.5;
	B2.at(0,1) = 2.0;
	B2.at(1,0) = -1.0;
	B2.at(1,1) = 2.5;
	B2.at(2,0) = -2.0;
	B2.at(2,1) = 5.0;
	A2.initComputation();
	B2.initComputation();
	A2.elementwiseMultiplicationWithClippedDerivative(B2, -1.5, 3.0);
	A2.finishComputation();
	EXPECT_EQ(1.0f, A2.at(0,0));
	EXPECT_EQ(-3.0f, A2.at(0,1));
	EXPECT_EQ(0.0f, A2.at(1,0));
	EXPECT_EQ(2.0f, A2.at(1,1));
	EXPECT_EQ(0.0f, A2.at(2,0));
	EXPECT_EQ(0.0f, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, multiplicationWithSoftmaxDerivative)
{
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	A.resize(3,2);
	B.resize(3,2);
	A.at(0,0) = 0;
	A.at(0,1) = 1;
	A.at(1,0) = -1;
	A.at(1,1) = -2;
	A.at(2,0) = 3;
	A.at(2,1) = -2;
	B.at(0,0) = 1;
	B.at(0,1) = -4;
	B.at(1,0) = -3;
	B.at(1,1) = 2;
	B.at(2,0) = 4;
	B.at(2,1) = 3;
	A.initComputation();
	B.initComputation();
	A.multiplicationWithSoftmaxDerivative(B);
	A.finishComputation();
	EXPECT_EQ(-15.0, A.at(0,0));
	EXPECT_EQ(-60.0, A.at(0,1));
	EXPECT_EQ(48.0, A.at(1,0));
	EXPECT_EQ(24.0, A.at(1,1));
	EXPECT_EQ(-48.0, A.at(2,0));
	EXPECT_EQ(36.0, A.at(2,1));

	Math::CudaMatrix<f32> A2;
	Math::CudaMatrix<f32> B2;
	A2.resize(3,2);
	B2.resize(3,2);
	A2.at(0,0) = 0;
	A2.at(0,1) = 1;
	A2.at(1,0) = -1;
	A2.at(1,1) = -2;
	A2.at(2,0) = 3;
	A2.at(2,1) = -2;
	B2.at(0,0) = 1;
	B2.at(0,1) = -4;
	B2.at(1,0) = -3;
	B2.at(1,1) = 2;
	B2.at(2,0) = 4;
	B2.at(2,1) = 3;
	A2.initComputation();
	B2.initComputation();
	A2.multiplicationWithSoftmaxDerivative(B2);
	A2.finishComputation();
	EXPECT_EQ(-15.0f, A2.at(0,0));
	EXPECT_EQ(-60.0f, A2.at(0,1));
	EXPECT_EQ(48.0f, A2.at(1,0));
	EXPECT_EQ(24.0f, A2.at(1,1));
	EXPECT_EQ(-48.0f, A2.at(2,0));
	EXPECT_EQ(36.0f, A2.at(2,1));
}

TEST_F(Test, TestCudaMatrix, addToColumn)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(2);
	A.at(0,0) = 1.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = -2.0;
	A.at(1,1) = 3.5;
	A.at(1,2) = 1.0;
	b.at(0) = 2.0;
	b.at(1) = -1.0;
	A.initComputation();
	b.initComputation();
	A.addToColumn(b, 1);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(8.0, A.at(0,1));
	EXPECT_EQ(0.0, A.at(0,2));
	EXPECT_EQ(-2.0, A.at(1,0));
	EXPECT_EQ(2.5, A.at(1,1));
	EXPECT_EQ(1.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = -2.0;
	A2.at(1,1) = 3.5;
	A2.at(1,2) = 1.0;
	b2.at(0) = 2.0;
	b2.at(1) = -1.0;
	A2.initComputation();
	b2.initComputation();
	A2.addToColumn(b2, 1);
	A2.finishComputation();
	EXPECT_EQ((f32)1.0, A2.at(0,0));
	EXPECT_EQ((f32)8.0, A2.at(0,1));
	EXPECT_EQ((f32)0.0, A2.at(0,2));
	EXPECT_EQ((f32)-2.0, A2.at(1,0));
	EXPECT_EQ((f32)2.5, A2.at(1,1));
	EXPECT_EQ((f32)1.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, addToRow)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(3);
	A.at(0,0) = 1.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = -2.0;
	A.at(1,1) = 3.5;
	A.at(1,2) = 1.0;
	b.at(0) = 2.0;
	b.at(1) = -1.0;
	b.at(2) = 4.0;
	A.initComputation();
	b.initComputation();
	A.addToRow(b,1);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(6.0, A.at(0,1));
	EXPECT_EQ(0.0, A.at(0,2));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(2.5, A.at(1,1));
	EXPECT_EQ(5.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(3);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = -2.0;
	A2.at(1,1) = 3.5;
	A2.at(1,2) = 1.0;
	b2.at(0) = 2.0;
	b2.at(1) = -1.0;
	b2.at(2) = 4.0;
	A2.initComputation();
	b2.initComputation();
	A2.addToRow(b2,1);
	A2.finishComputation();
	EXPECT_EQ((f32)1.0, A2.at(0,0));
	EXPECT_EQ((f32)6.0, A2.at(0,1));
	EXPECT_EQ((f32)0.0, A2.at(0,2));
	EXPECT_EQ((f32)0.0, A2.at(1,0));
	EXPECT_EQ((f32)2.5, A2.at(1,1));
	EXPECT_EQ((f32)5.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, multiplyColumnByScalar)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	A.initComputation();
	A.multiplyColumnByScalar(1, -2.0);
	A.finishComputation();
	EXPECT_EQ(2.0, A.at(0,0));
	EXPECT_EQ(-2.0, A.at(1,0));
	EXPECT_EQ(-8.0, A.at(0,1));
	EXPECT_EQ(16.0, A.at(1,1));
	EXPECT_EQ(5.0, A.at(0,2));
	EXPECT_EQ(-6.0, A.at(1,2));

	Math::CudaMatrix<f64> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	A2.initComputation();
	A2.multiplyColumnByScalar(1, -2.0);
	A2.finishComputation();
	EXPECT_EQ(2.0, A2.at(0,0));
	EXPECT_EQ(-2.0, A2.at(1,0));
	EXPECT_EQ(-8.0, A2.at(0,1));
	EXPECT_EQ(16.0, A2.at(1,1));
	EXPECT_EQ(5.0, A2.at(0,2));
	EXPECT_EQ(-6.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, multiplyRowByScalar)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	A.initComputation();
	A.multiplyRowByScalar(1, -2.0);
	A.finishComputation();
	EXPECT_EQ(2.0, A.at(0,0));
	EXPECT_EQ(4.0, A.at(1,0));
	EXPECT_EQ(4.0, A.at(0,1));
	EXPECT_EQ(16.0, A.at(1,1));
	EXPECT_EQ(5.0, A.at(0,2));
	EXPECT_EQ(12.0, A.at(1,2));

	Math::CudaMatrix<f64> A2;
	A2.resize(2,3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	A2.initComputation();
	A2.multiplyRowByScalar(1, -2.0);
	A2.finishComputation();
	EXPECT_EQ(2.0, A2.at(0,0));
	EXPECT_EQ(4.0, A2.at(1,0));
	EXPECT_EQ(4.0, A2.at(0,1));
	EXPECT_EQ(16.0, A2.at(1,1));
	EXPECT_EQ(5.0, A2.at(0,2));
	EXPECT_EQ(12.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, addToAllChannels)
{
	Math::CudaMatrix<f64> m_64(12, 1);
	Math::CudaMatrix<f32> m_32(12, 1);

	Math::CudaVector<f64> v_64(3);
	Math::CudaVector<f32> v_32(3);

	for(u32 i=0; i<m_64.nRows(); i++) {
		for(u32 j=0; j<m_64.nColumns(); j++) {
			m_64.at(i,j) = i;
			m_32.at(i,j) = i;
		}
	}
	for(u32 i=0; i<3; i++) {
		v_64.at(i) = i;
		v_32.at(i) = i;
	}

	m_64.initComputation();
	m_32.initComputation();
	v_64.initComputation();
	v_32.initComputation();

	m_64.addToAllChannels(v_64, 3);
	m_32.addToAllChannels(v_32, 3);

	m_64.finishComputation();
	m_32.finishComputation();
	v_32.finishComputation(false);
	v_64.finishComputation(false);

	for(u32 i=0; i<m_32.nRows(); i++) {
		for(u32 j=0; j<m_32.nColumns(); j++) {
			EXPECT_EQ(m_32.at(i,j), (float)(i + (i/4)));
			EXPECT_DOUBLE_EQ(m_64.at(i,j), i + (i/4), 0.0000001);
		}
	}
}

TEST_F(Test, TestCudaMatrix, addToAllColumns)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(2);
	A.at(0,0) = 1.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = -2.0;
	A.at(1,1) = 3.5;
	A.at(1,2) = 1.0;
	b.at(0) = 2.0;
	b.at(1) = -1.0;
	A.initComputation();
	b.initComputation();
	A.addToAllColumns(b);
	A.finishComputation();
	EXPECT_EQ(3.0, A.at(0,0));
	EXPECT_EQ(8.0, A.at(0,1));
	EXPECT_EQ(2.0, A.at(0,2));
	EXPECT_EQ(-3.0, A.at(1,0));
	EXPECT_EQ(2.5, A.at(1,1));
	EXPECT_EQ(0.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(2);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = -2.0;
	A2.at(1,1) = 3.5;
	A2.at(1,2) = 1.0;
	b2.at(0) = 2.0;
	b2.at(1) = -1.0;
	A2.initComputation();
	b2.initComputation();
	A2.addToAllColumns(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)3.0, A2.at(0,0));
	EXPECT_EQ((f32)8.0, A2.at(0,1));
	EXPECT_EQ((f32)2.0, A2.at(0,2));
	EXPECT_EQ((f32)-3.0, A2.at(1,0));
	EXPECT_EQ((f32)2.5, A2.at(1,1));
	EXPECT_EQ((f32)0.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, addToAllRows)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(3);
	A.at(0,0) = 1.0;
	A.at(0,1) = 6.0;
	A.at(0,2) = 0.0;
	A.at(1,0) = -2.0;
	A.at(1,1) = 3.5;
	A.at(1,2) = 1.0;
	b.at(0) = 2.0;
	b.at(1) = -1.0;
	b.at(2) = 4.0;
	A.initComputation();
	b.initComputation();
	A.addToAllRows(b);
	A.finishComputation();
	EXPECT_EQ(3.0, A.at(0,0));
	EXPECT_EQ(5.0, A.at(0,1));
	EXPECT_EQ(4.0, A.at(0,2));
	EXPECT_EQ(0.0, A.at(1,0));
	EXPECT_EQ(2.5, A.at(1,1));
	EXPECT_EQ(5.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(3);
	A2.at(0,0) = 1.0;
	A2.at(0,1) = 6.0;
	A2.at(0,2) = 0.0;
	A2.at(1,0) = -2.0;
	A2.at(1,1) = 3.5;
	A2.at(1,2) = 1.0;
	b2.at(0) = 2.0;
	b2.at(1) = -1.0;
	b2.at(2) = 4.0;
	A2.initComputation();
	b2.initComputation();
	A2.addToAllRows(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)3.0, A2.at(0,0));
	EXPECT_EQ((f32)5.0, A2.at(0,1));
	EXPECT_EQ((f32)4.0, A2.at(0,2));
	EXPECT_EQ((f32)0.0, A2.at(1,0));
	EXPECT_EQ((f32)2.5, A2.at(1,1));
	EXPECT_EQ((f32)5.0, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, multiplyColumnsByScalars)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	b.at(0) = 2.0;
	b.at(1) = -2.0;
	b.at(2) = -1.0;
	A.initComputation();
	b.initComputation();
	A.multiplyColumnsByScalars(b);
	A.finishComputation();
	EXPECT_EQ(4.0, A.at(0,0));
	EXPECT_EQ(-4.0, A.at(1,0));
	EXPECT_EQ(-8.0, A.at(0,1));
	EXPECT_EQ(16.0, A.at(1,1));
	EXPECT_EQ(-5.0, A.at(0,2));
	EXPECT_EQ(6.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	b2.at(0) = 2.0;
	b2.at(1) = -2.0;
	b2.at(2) = -1.0;
	A2.initComputation();
	b2.initComputation();
	A2.multiplyColumnsByScalars(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)4.0f, A2.at(0,0));
	EXPECT_EQ((f32)-4.0f, A2.at(1,0));
	EXPECT_EQ((f32)-8.0f, A2.at(0,1));
	EXPECT_EQ((f32)16.0f, A2.at(1,1));
	EXPECT_EQ((f32)-5.0f, A2.at(0,2));
	EXPECT_EQ((f32)6.0f, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, divideColumnsByScalars)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(3);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	b.at(0) = 2.0;
	b.at(1) = -2.0;
	b.at(2) = -1.0;
	A.initComputation();
	b.initComputation();
	A.divideColumnsByScalars(b);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(-1.0, A.at(1,0));
	EXPECT_EQ(-2.0, A.at(0,1));
	EXPECT_EQ(4.0, A.at(1,1));
	EXPECT_EQ(-5.0, A.at(0,2));
	EXPECT_EQ(6.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(3);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	b2.at(0) = 2.0;
	b2.at(1) = -2.0;
	b2.at(2) = -1.0;
	A2.initComputation();
	b2.initComputation();
	A2.divideColumnsByScalars(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)1.0f, A2.at(0,0));
	EXPECT_EQ((f32)-1.0f, A2.at(1,0));
	EXPECT_EQ((f32)-2.0f, A2.at(0,1));
	EXPECT_EQ((f32)4.0f, A2.at(1,1));
	EXPECT_EQ((f32)-5.0f, A2.at(0,2));
	EXPECT_EQ((f32)6.0f, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, multiplyRowsByScalars)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	b.at(0) = 2.0;
	b.at(1) = -2.0;
	A.initComputation();
	b.initComputation();
	A.multiplyRowsByScalars(b);
	A.finishComputation();
	EXPECT_EQ(4.0, A.at(0,0));
	EXPECT_EQ(4.0, A.at(1,0));
	EXPECT_EQ(8.0, A.at(0,1));
	EXPECT_EQ(16.0, A.at(1,1));
	EXPECT_EQ(10.0, A.at(0,2));
	EXPECT_EQ(12.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(2);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	b2.at(0) = 2.0;
	b2.at(1) = -2.0;
	A2.initComputation();
	b2.initComputation();
	A2.multiplyRowsByScalars(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)4.0f, A2.at(0,0));
	EXPECT_EQ((f32)4.0f, A2.at(1,0));
	EXPECT_EQ((f32)8.0f, A2.at(0,1));
	EXPECT_EQ((f32)16.0f, A2.at(1,1));
	EXPECT_EQ((f32)10.0f, A2.at(0,2));
	EXPECT_EQ((f32)12.0f, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, divideRowsByScalars)
{
	Math::CudaMatrix<f64> A;
	Math::CudaVector<f64> b;
	A.resize(2,3);
	b.resize(2);
	A.at(0,0) = 2.0;
	A.at(1,0) = -2.0;
	A.at(0,1) = 4.0;
	A.at(1,1) = -8.0;
	A.at(0,2) = 5.0;
	A.at(1,2) = -6.0;
	b.at(0) = 2.0;
	b.at(1) = -2.0;
	A.initComputation();
	b.initComputation();
	A.divideRowsByScalars(b);
	A.finishComputation();
	EXPECT_EQ(1.0, A.at(0,0));
	EXPECT_EQ(1.0, A.at(1,0));
	EXPECT_EQ(2.0, A.at(0,1));
	EXPECT_EQ(4.0, A.at(1,1));
	EXPECT_EQ(2.5, A.at(0,2));
	EXPECT_EQ(3.0, A.at(1,2));

	Math::CudaMatrix<f32> A2;
	Math::CudaVector<f32> b2;
	A2.resize(2,3);
	b2.resize(2);
	A2.at(0,0) = 2.0;
	A2.at(1,0) = -2.0;
	A2.at(0,1) = 4.0;
	A2.at(1,1) = -8.0;
	A2.at(0,2) = 5.0;
	A2.at(1,2) = -6.0;
	b2.at(0) = 2.0;
	b2.at(1) = -2.0;
	A2.initComputation();
	b2.initComputation();
	A2.divideRowsByScalars(b2);
	A2.finishComputation();
	EXPECT_EQ((f32)1.0f, A2.at(0,0));
	EXPECT_EQ((f32)1.0f, A2.at(1,0));
	EXPECT_EQ((f32)2.0f, A2.at(0,1));
	EXPECT_EQ((f32)4.0f, A2.at(1,1));
	EXPECT_EQ((f32)2.5f, A2.at(0,2));
	EXPECT_EQ((f32)3.0f, A2.at(1,2));
}

TEST_F(Test, TestCudaMatrix, softmax)
{
	Math::CudaMatrix<f64> A;
	A.resize(3,2);
	A.at(0,0) = 4.05; A.at(1,0) = 4.9; A.at(2,0) = 4.8;
	A.at(0,1) = 1.7; A.at(1,1) = 2.1; A.at(2,1) = 1.95;
	A.initComputation();
	A.softmax();
	A.finishComputation();
	EXPECT_DOUBLE_EQ(0.18326272967482829, A.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.42877006855907612, A.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.38796720176609562, A.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.26484102115311464, A.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.39509637630475053, A.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.34006260254213494, A.at(2,1), 0.000001);
}

TEST_F(Test, TestCudaMatrix, chiSquareFeatureMap)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,2);
	A.at(0,0) = 0.3; A.at(1,0) = 0.7;
	A.at(0,1) = 0.9; A.at(1,1) = 0.1;
	Math::CudaMatrix<f64> B;
	B.resize(10, 2);
	A.initComputation();
	B.initComputation();
	B.chiSquareFeatureMap(A, 2, 0.5);
	B.finishComputation();
	EXPECT_DOUBLE_EQ(0.387298, B.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.284993, B.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.195806, B.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.0576973, B.at(3,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.15017, B.at(4,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.591608, B.at(5,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.519805, B.at(6,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0936961, B.at(7,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.230271, B.at(8,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0858017, B.at(9,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.67082, B.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.598071, B.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0315357, B.at(2,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.277095, B.at(3,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0293033, B.at(4,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.223607, B.at(5,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.0813123, B.at(6,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.182324, B.at(7,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0620625, B.at(8,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.0691008, B.at(9,1), 0.000001);
}

TEST_F(Test, TestCudaMatrix, multiplicationWithChiSquareFeatureMapDerivative)
{
	Math::CudaMatrix<f64> A;
	A.resize(2,2);
	A.at(0,0) = 0.3; A.at(1,0) = 0.7;
	A.at(0,1) = 0.9; A.at(1,1) = 0.1;
	Math::CudaMatrix<f64> B;
	B.resize(10, 2);
	Math::CudaMatrix<f64> C;
	C.resize(10, 2);
	A.initComputation();
	B.initComputation();
	C.initComputation();
	B.chiSquareFeatureMap(A, 2, 0.5);
	C.fill(1.0);
	C.elementwiseMultiplicationWithApproximateFeatureMapDerivative(B, 2, 0.5, 1.0);
	C.finishComputation();
	EXPECT_DOUBLE_EQ(0.6454966666666666, C.at(0,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.8013316666666667, C.at(1,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.148645, C.at(2,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.5967288333333334, C.at(3,0), 0.000001);
	EXPECT_DOUBLE_EQ(-0.057959, C.at(4,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.4225771428571429, C.at(5,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.4382150714285714, C.at(6,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.3043635, C.at(7,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.28705314285714284, C.at(8,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.2676716428571429, C.at(9,0), 0.000001);
	EXPECT_DOUBLE_EQ(0.3726777777777778, C.at(0,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.3497815, C.at(1,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.31474183333333333, C.at(2,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.18650088888888888, C.at(3,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.2916037222222222, C.at(4,1), 0.000001);
	EXPECT_DOUBLE_EQ(1.11803398874989, C.at(5,1), 0.000001);
	EXPECT_DOUBLE_EQ(1.3181815000000001, C.at(6,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.5050585, C.at(7,1), 0.000001);
	EXPECT_DOUBLE_EQ(0.3806955000000001, C.at(8,1), 0.000001);
	EXPECT_DOUBLE_EQ(-0.966128879489055, C.at(9,1), 0.000001);
}

TEST_F(Test, TestCudaMatrix, nClassificationErrors)
{
	Math::CudaMatrix<f32> X(5,3);
	X.at(0,0) = 4.0f;
	X.at(1,0) = 5.0f; // error
	X.at(2,0) = 2.0f;
	X.at(3,0) = 0.0f;
	X.at(4,0) = -10.0f;

	X.at(0,1) = 1.0f;
	X.at(1,1) = 3.0f; // error
	X.at(2,1) = 0.0f;
	X.at(3,1) = 1.0f;
	X.at(4,1) = 4.0f;

	X.at(0,2) = -10.0f;
	X.at(1,2) = 0.0f;
	X.at(2,2) = 1.0f;
	X.at(3,2) = 5.0f;
	X.at(4,2) = 8.0f; // correct

	Math::CudaMatrix<f32> targets(5,3);
	targets.initComputation();
	targets.setToZero();
	targets.finishComputation();
	targets.at(0,0) = 1.0;
	targets.at(1,1) = 1.0;
	targets.at(4,2) = 1.0;

	X.initComputation();
	targets.initComputation();

	u32 result = X.nClassificationErrors(targets);
	EXPECT_EQ(2u, result);

	Math::CudaMatrix<f64> X2(5,3);
	X2.at(0,0) = 4.0f;
	X2.at(1,0) = 5.0f; // error
	X2.at(2,0) = 2.0f;
	X2.at(3,0) = 0.0f;
	X2.at(4,0) = -10.0f;

	X2.at(0,1) = 1.0f;
	X2.at(1,1) = 3.0f; // error
	X2.at(2,1) = 0.0f;
	X2.at(3,1) = 1.0f;
	X2.at(4,1) = 4.0f;

	X2.at(0,2) = -10.0f;
	X2.at(1,2) = 0.0f;
	X2.at(2,2) = 1.0f;
	X2.at(3,2) = 5.0f;
	X2.at(4,2) = 8.0f; // correct

	Math::CudaMatrix<f64> targets2(5,3);
	targets2.initComputation();
	targets2.setToZero();
	targets2.finishComputation();
	targets2.at(0,0) = 1.0;
	targets2.at(1,1) = 1.0;
	targets2.at(4,2) = 1.0;

	X2.initComputation();
	targets2.initComputation();

	result = X2.nClassificationErrors(targets2);
	EXPECT_EQ(2u, result);
}

TEST_F(Test, TestCudaMatrix, crossEntropyObjectiveFunction)
{
	// values do not make sense, but that does not matter here
	Math::CudaMatrix<f64> X(5,3);
	X.at(0,0) = 0.1f;
	X.at(0,1) = 0.1f;
	X.at(0,2) = 0.2f;
	X.at(1,0) = 0.5f;
	X.at(1,1) = 0.3f;
	X.at(1,2) = 0.0f;
	X.at(2,0) = 0.2f;
	X.at(2,1) = 0.0f;
	X.at(2,2) = 0.1f;
	X.at(3,0) = 0.0f;
	X.at(3,1) = 0.1f;
	X.at(3,2) = 0.5f;
	X.at(4,0) = 0.2f;
	X.at(4,1) = 0.4f;
	X.at(4,2) = 0.8f;

	Math::CudaMatrix<f64> targets(5,3);
	targets.initComputation();
	targets.setToZero();
	targets.finishComputation();
	targets.at(0,0) = 1.0;
	targets.at(1,1) = 1.0;
	targets.at(4,2) = 1.0;

	X.initComputation();
	targets.initComputation();

	f32 result = X.crossEntropyObjectiveFunction(targets);
	f32 checkVal = -std::log(0.3) - std::log(0.8) - std::log(0.1);
	EXPECT_DOUBLE_EQ(checkVal, result, 0.000001);

	Math::CudaMatrix<f32> X2(5,3);
	X2.at(0,0) = 0.1f;
	X2.at(0,1) = 0.1f;
	X2.at(0,2) = 0.2f;
	X2.at(1,0) = 0.5f;
	X2.at(1,1) = 0.3f;
	X2.at(1,2) = 0.0f;
	X2.at(2,0) = 0.2f;
	X2.at(2,1) = 0.0f;
	X2.at(2,2) = 0.1f;
	X2.at(3,0) = 0.0f;
	X2.at(3,1) = 0.1f;
	X2.at(3,2) = 0.5f;
	X2.at(4,0) = 0.2f;
	X2.at(4,1) = 0.4f;
	X2.at(4,2) = 0.8f;

	Math::CudaMatrix<f32> targets2(5,3);
	targets2.initComputation();
	targets2.setToZero();
	targets2.finishComputation();
	targets2.at(0,0) = 1.0;
	targets2.at(1,1) = 1.0;
	targets2.at(4,2) = 1.0;

	X2.initComputation();
	targets2.initComputation();

	result = X2.crossEntropyObjectiveFunction(targets2);
	checkVal = -std::log(0.3) - std::log(0.8) - std::log(0.1);
	EXPECT_DOUBLE_EQ(checkVal, result, 0.000001);
}

TEST_F(Test, TestCudaMatrix, smoothedL1ObjectiveFunction)
{
	// values do not make sense, but that does not matter here
	Math::CudaMatrix<f64> X(2,3);
	X.at(0,0) = 0.1f;
	X.at(0,1) = 2.0f;
	X.at(0,2) = 0.0f;
	X.at(1,0) = -0.5f;
	X.at(1,1) = -3.0f;
	X.at(1,2) = 0.9f;

	Math::CudaMatrix<f64> targets(2,3);
	targets.initComputation();
	targets.setToZero();
	targets.finishComputation();
	targets.at(0,0) = 1.0;
	targets.at(1,1) = 1.0;

	X.initComputation();
	targets.initComputation();

	f32 result = X.smoothedL1ObjectiveFunction(targets);
	f32 checkVal = 5.935;
	EXPECT_EQ(checkVal, result);
}

TEST_F(Test, TestCudaMatrix, weightedSmoothedL1ObjectiveFunction)
{
	// values do not make sense, but that does not matter here
	Math::CudaMatrix<f64> X(2,3);
	X.at(0,0) = 0.1f;
	X.at(0,1) = 2.0f;
	X.at(0,2) = 0.0f;
	X.at(1,0) = -0.5f;
	X.at(1,1) = -3.0f;
	X.at(1,2) = 0.9f;

	Math::CudaMatrix<f64> targets(2,3);
	targets.initComputation();
	targets.setToZero();
	targets.finishComputation();
	targets.at(0,0) = 1.0;
	targets.at(1,1) = 1.0;

	Math::CudaVector<f64> weights(3);
	weights.at(0) = 1.0f;
	weights.at(1) = 2.0f;
	weights.at(2) = 0.5f;
	weights.initComputation();

	X.initComputation();
	targets.initComputation();

	f32 result = X.weightedSmoothedL1ObjectiveFunction(targets, weights);
	f32 checkVal = 10.7325;
	EXPECT_EQ(checkVal, result);
}

TEST_F(Test, TestCudaMatrix, addSummedNeighborsInARow)
{
	Math::CudaMatrix<f64> A(2,2);
	Math::CudaMatrix<f64> B(6,2);
	A.at(0,0) = 0;
	A.at(1,0) = -1;
	A.at(0,1) = 1;
	A.at(1,1) = 0;
	for (u32 i = 0; i < 6; i++) {
		B.at(i, 0) = i;
		B.at(i, 1) = i + 1;
	}
	A.initComputation();
	B.initComputation();
	A.addSummedNeighborsInARow(B, 3);
	A.finishComputation();
	EXPECT_EQ(3.0, A.at(0,0));
	EXPECT_EQ(11.0, A.at(1,0));
	EXPECT_EQ(7.0, A.at(0,1));
	EXPECT_EQ(15.0, A.at(1,1));
}

TEST_F(Test, TestCudaMatrix, clone)
{
	Math::CudaMatrix<f64> A(3,2);
	Math::CudaMatrix<f64> B(9,2);
	for (u32 i = 0; i < 3; i++) {
		A.at(i, 0) = i;
		A.at(i, 1) = i + 3;
	}
	A.initComputation();
	B.initComputation();
	B.setToZero();
	B.clone(A, 3);
	B.finishComputation();
	for (u32 i = 0; i < 3; i++) {
		for (u32 n = 0; n < 3; n++) {
			EXPECT_EQ((f64)i, B.at(n * 3 + i, 0));
			EXPECT_EQ((f64)i + 3, B.at(n * 3 + i, 1));
		}
	}
}

TEST_F(Test, TestCudaMatrix, cloneElementwise)
{
	Math::CudaMatrix<f64> A(3,2);
	Math::CudaMatrix<f64> B(9,2);
	for (u32 i = 0; i < 3; i++) {
		A.at(i, 0) = i;
		A.at(i, 1) = i + 3;
	}
	A.initComputation();
	B.initComputation();
	B.setToZero();
	B.cloneElementwise(A, 3);
	B.finishComputation();
	for (u32 i = 0; i < 3; i++) {
		for (u32 n = 0; n < 3; n++) {
			EXPECT_EQ((f64)i, B.at(i * 3 + n, 0));
			EXPECT_EQ((f64)i + 3, B.at(i * 3 + n, 1));
		}
	}
}

TEST_F(Test, TestCudaMatrix, addElementsByModuloIndex)
{
	Math::CudaMatrix<f64> A(6,2);
	Math::CudaMatrix<f64> B(3,2);
	for (u32 i = 0; i < 6; i++) {
		A.at(i, 0) = i;
		A.at(i, 1) = i + 6;
	}
	A.initComputation();
	B.initComputation();
	B.setToZero();
	B.addElementsByModuloIndex(A);
	B.finishComputation();
	EXPECT_EQ(3.0, B.at(0,0));
	EXPECT_EQ(5.0, B.at(1,0));
	EXPECT_EQ(7.0, B.at(2,0));
	EXPECT_EQ(15.0, B.at(0,1));
	EXPECT_EQ(17.0, B.at(1,1));
	EXPECT_EQ(19.0, B.at(2,1));
}

TEST_F(Test, TestCudaMatrix, setToSecondOrderFeatures){
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	B.resize(2,3);
	A.resize(5,3);

	B.at(0,0) = 1.0;
	B.at(1,0) = 2.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = -2.0;
	B.at(0,2) = 1.0;
	B.at(1,2) = 3.0;
	A.initComputation();
	B.initComputation();
	A.setToSecondOrderFeatures(B);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 1.0);
	EXPECT_EQ(A.at(1,0), 2.0);
	EXPECT_EQ(A.at(2,0), 1.0);
	EXPECT_EQ(A.at(3,0), 2.0);
	EXPECT_EQ(A.at(4,0), 4.0);

	EXPECT_EQ(A.at(0,1), -1.0);
	EXPECT_EQ(A.at(1,1), -2.0);
	EXPECT_EQ(A.at(2,1), 1.0);
	EXPECT_EQ(A.at(3,1), 2.0);
	EXPECT_EQ(A.at(4,1), 4.0);

	EXPECT_EQ(A.at(0,2), 1.0);
	EXPECT_EQ(A.at(1,2), 3.0);
	EXPECT_EQ(A.at(2,2), 1.0);
	EXPECT_EQ(A.at(3,2), 3.0);
	EXPECT_EQ(A.at(4,2), 9.0);
}

TEST_F(Test, TestCudaMatrix, setToDiagonalSecondOrderFeatures){
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	B.resize(2,3);
	A.resize(4,3);

	B.at(0,0) = 1.0;
	B.at(1,0) = 2.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = -2.0;
	B.at(0,2) = 1.0;
	B.at(1,2) = 3.0;
	A.initComputation();
	B.initComputation();
	A.setToDiagonalSecondOrderFeatures(B);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 1.0);
	EXPECT_EQ(A.at(1,0), 2.0);
	EXPECT_EQ(A.at(2,0), 1.0);
	EXPECT_EQ(A.at(3,0), 4.0);

	EXPECT_EQ(A.at(0,1), -1.0);
	EXPECT_EQ(A.at(1,1), -2.0);
	EXPECT_EQ(A.at(2,1), 1.0);
	EXPECT_EQ(A.at(3,1), 4.0);

	EXPECT_EQ(A.at(0,2), 1.0);
	EXPECT_EQ(A.at(1,2), 3.0);
	EXPECT_EQ(A.at(2,2), 1.0);
	EXPECT_EQ(A.at(3,2), 9.0);
}

TEST_F(Test, TestCudaMatrix, setToThirdOrderFeatures){
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	B.resize(2,3);
	A.resize(9,3);

	B.at(0,0) = 1.0;
	B.at(1,0) = 2.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = -2.0;
	B.at(0,2) = 1.0;
	B.at(1,2) = 3.0;
	A.initComputation();
	B.initComputation();
	A.setToThirdOrderFeatures(B);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 1.0);
	EXPECT_EQ(A.at(1,0), 2.0);
	EXPECT_EQ(A.at(2,0), 1.0);
	EXPECT_EQ(A.at(3,0), 2.0);
	EXPECT_EQ(A.at(4,0), 4.0);
	EXPECT_EQ(A.at(5,0), 1.0);
	EXPECT_EQ(A.at(6,0), 2.0);
	EXPECT_EQ(A.at(7,0), 4.0);
	EXPECT_EQ(A.at(8,0), 8.0);

	EXPECT_EQ(A.at(0,1), -1.0);
	EXPECT_EQ(A.at(1,1), -2.0);
	EXPECT_EQ(A.at(2,1), 1.0);
	EXPECT_EQ(A.at(3,1), 2.0);
	EXPECT_EQ(A.at(4,1), 4.0);
	EXPECT_EQ(A.at(5,1), -1.0);
	EXPECT_EQ(A.at(6,1), -2.0);
	EXPECT_EQ(A.at(7,1), -4.0);
	EXPECT_EQ(A.at(8,1), -8.0);

	EXPECT_EQ(A.at(0,2), 1.0);
	EXPECT_EQ(A.at(1,2), 3.0);
	EXPECT_EQ(A.at(2,2), 1.0);
	EXPECT_EQ(A.at(3,2), 3.0);
	EXPECT_EQ(A.at(4,2), 9.0);
	EXPECT_EQ(A.at(5,2), 1.0);
	EXPECT_EQ(A.at(6,2), 3.0);
	EXPECT_EQ(A.at(7,2), 9.0);
	EXPECT_EQ(A.at(8,2), 27.0);
}

TEST_F(Test, TestCudaMatrix, setToDiagonalThirdOrderFeatures){
	Math::CudaMatrix<f64> A;
	Math::CudaMatrix<f64> B;
	B.resize(2,3);
	A.resize(6,3);

	B.at(0,0) = 1.0;
	B.at(1,0) = 2.0;
	B.at(0,1) = -1.0;
	B.at(1,1) = -2.0;
	B.at(0,2) = 1.0;
	B.at(1,2) = 3.0;
	A.initComputation();
	B.initComputation();
	A.setToDiagonalThirdOrderFeatures(B);
	A.finishComputation();
	EXPECT_EQ(A.at(0,0), 1.0);
	EXPECT_EQ(A.at(1,0), 2.0);
	EXPECT_EQ(A.at(2,0), 1.0);
	EXPECT_EQ(A.at(3,0), 4.0);
	EXPECT_EQ(A.at(4,0), 1.0);
	EXPECT_EQ(A.at(5,0), 8.0);

	EXPECT_EQ(A.at(0,1), -1.0);
	EXPECT_EQ(A.at(1,1), -2.0);
	EXPECT_EQ(A.at(2,1), 1.0);
	EXPECT_EQ(A.at(3,1), 4.0);
	EXPECT_EQ(A.at(4,1), -1.0);
	EXPECT_EQ(A.at(5,1), -8.0);

	EXPECT_EQ(A.at(0,2), 1.0);
	EXPECT_EQ(A.at(1,2), 3.0);
	EXPECT_EQ(A.at(2,2), 1.0);
	EXPECT_EQ(A.at(3,2), 9.0);
	EXPECT_EQ(A.at(4,2), 1.0);
	EXPECT_EQ(A.at(5,2), 27.0);
}

TEST_F(Test, TestCudaMatrix, gaussianMixturePosteriors){
	Math::CudaMatrix<f64> A(2,2);
	Math::CudaMatrix<f64> B(3,2);
	Math::CudaMatrix<f64> means(3,2);
	Math::CudaMatrix<f64> variances(3,2);
	Math::CudaVector<f64> weights(3);
	A.at(0,0) = 1.0;
	A.at(1,0) = -1.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	means.at(0,0) = 0;
	means.at(1,0) = 1;
	means.at(2,0) = -1;
	means.at(0,1) = 2;
	means.at(1,1) = -1;
	means.at(2,1) = 1;
	variances.at(0,0) = 1;
	variances.at(1,0) = 0.5;
	variances.at(2,0) = 2;
	variances.at(0,1) = 0.5;
	variances.at(1,1) = 2;
	variances.at(2,1) = 1;
	weights.at(0) = 0.3;
	weights.at(1) = 0.5;
	weights.at(2) = 0.2;

	A.initComputation();
	B.initComputation();
	means.initComputation();
	variances.initComputation();
	weights.initComputation();
	B.gaussianMixturePosteriors(A, means, variances, weights);
	B.finishComputation();

	EXPECT_DOUBLE_EQ(B.at(0,0), 6.2628e-05, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(1,0), 0.986052, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(2,0), 0.0138855, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(0,1), 0.831151, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(1,1), 0.0379801, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(2,1), 0.130869, 0.0001);
}

TEST_F(Test, TestCudaMatrix, fisherEncoding){
	Math::CudaMatrix<f64> A(2,2);
	Math::CudaMatrix<f64> B(12,2);
	Math::CudaMatrix<f64> means(3,2);
	Math::CudaMatrix<f64> variances(3,2);
	Math::CudaVector<f64> weights(3);
	A.at(0,0) = 1.0;
	A.at(1,0) = -1.0;
	A.at(0,1) = 0.0;
	A.at(1,1) = 2.0;
	means.at(0,0) = 0;
	means.at(1,0) = 1;
	means.at(2,0) = -1;
	means.at(0,1) = 2;
	means.at(1,1) = -1;
	means.at(2,1) = 1;
	variances.at(0,0) = 1;
	variances.at(1,0) = 0.5;
	variances.at(2,0) = 2;
	variances.at(0,1) = 0.5;
	variances.at(1,1) = 2;
	variances.at(2,1) = 1;
	weights.at(0) = 0.3;
	weights.at(1) = 0.5;
	weights.at(2) = 0.2;

	A.initComputation();
	B.initComputation();
	means.initComputation();
	variances.initComputation();
	weights.initComputation();
	B.fisherEncoding(A, means, variances, weights);
	B.finishComputation();

	EXPECT_DOUBLE_EQ(B.at(0,0), 0.000114343, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(1,0), -0.000485114, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(2,0), 0.0, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(3,0), 0.0, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(4,0), 0.0439098, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(5,0), -0.0620978, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(6,0), 0.0, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(7,0), 0.00137449, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(8,0), -0.986052, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(9,0), -0.986052, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(10,0), 0.0219549, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(11,0), 0.0658647, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(0,1), 0.0, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(1,1), 0.0, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(2,1), -0.0759603, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(3,1), 0.11394, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(4,1), 0.206922, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(5,1), 0.292633, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(6,1), -1.07301, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(7,1), -1.07301, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(8,1), 0.0379801, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(9,1), 0.13293, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(10,1), -0.103461, 0.0001);
	EXPECT_DOUBLE_EQ(B.at(11,1), 0.0, 0.0001);
}

TEST_F(Test, TestCudaMatrix, isFinite)
{
	Math::CudaMatrix<f64> A;
	A.resize(4,3);
	A.setToZero();
	EXPECT_TRUE(A.isFinite());
	A.at(2,1) = std::log(0.0);
	EXPECT_FALSE(A.isFinite());
	A.at(2,1) = std::sqrt(-1.0);
	EXPECT_FALSE(A.isFinite());

	Math::CudaMatrix<f32> A2;
	A2.resize(4,3);
	A2.setToZero();
	EXPECT_TRUE(A2.isFinite());
	A2.at(2,1) = std::log(0.0f);
	EXPECT_FALSE(A2.isFinite());
	A2.at(2,1) = std::sqrt(-1.0f);
	EXPECT_FALSE(A2.isFinite());

}

TEST_F(Test, TestCudaMatrix, rpropUpdate)
{
	Math::CudaMatrix<f64> weights;
	Math::CudaMatrix<f64> newGradients;
	Math::CudaMatrix<f64> oldGradients;
	Math::CudaMatrix<f64> updateValues;
	weights.resize(2,3);
	newGradients.copyStructure(weights);
	oldGradients.copyStructure(weights);
	updateValues.copyStructure(weights);

	weights.at(0,0) = 3.0;
	weights.at(0,1) = 2.0;
	weights.at(0,2) = 1.0;
	weights.at(1,0) = -1.0;
	weights.at(1,1) = -2.0;
	weights.at(1,2) = -3.0;

	newGradients.at(0,0) = 2.0;
	newGradients.at(0,1) = -1.0;
	newGradients.at(0,2) = 1.0;
	newGradients.at(1,0) = 2.0;
	newGradients.at(1,1) = -2.5;
	newGradients.at(1,2) = 0.0;

	oldGradients.at(0,0) = -2.0;
	oldGradients.at(0,1) = 3.0;
	oldGradients.at(0,2) = 0.0;
	oldGradients.at(1,0) = 1.5;
	oldGradients.at(1,1) = -1.0;
	oldGradients.at(1,2) = 2.0;

	updateValues.at(0,0) = -1.0;
	updateValues.at(0,1) = 2.0;
	updateValues.at(0,2) = 1.5;
	updateValues.at(1,0) = -2.0;
	updateValues.at(1,1) = 1.0;
	updateValues.at(1,2) = -1.5;

	weights.initComputation();
	newGradients.initComputation();
	oldGradients.initComputation();
	updateValues.initComputation();

	weights.rpropUpdate(newGradients, oldGradients, updateValues, 1.2, 0.5, 100, -0.0001);

	weights.finishComputation();
	newGradients.finishComputation();
	oldGradients.finishComputation();
	updateValues.finishComputation();

	EXPECT_EQ(3.0001, weights.at(0,0));
	EXPECT_EQ(3.0, weights.at(0,1));
	EXPECT_EQ(-0.5, weights.at(0,2));
	EXPECT_EQ(1.4, weights.at(1,0));
	EXPECT_EQ(-0.8, weights.at(1,1));
	EXPECT_EQ(-3.0, weights.at(1,2));
}
