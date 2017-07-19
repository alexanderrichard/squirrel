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
 * TestMatrix.cc
 *
 *  Created on: 27.03.2014
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <Math/Matrix.hh>
#include <Math/Vector.hh>

class TestMatrix : public Test::Fixture
{
protected:
	Math::Matrix<f32> A_;
public:
	void setUp();
	void tearDown();
};

void TestMatrix::setUp()
{
	A_.resize(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			A_.at(i,j) = i+j;
		}
	}
}

void TestMatrix::tearDown()
{
}

TEST_F(Test, TestMatrix, resize)
{
	Math::Matrix<f64> A;
	EXPECT_TRUE(A.empty());
	A.resize(2,3);
	EXPECT_EQ(A.nRows(),2u);
	EXPECT_EQ(A.nColumns(),3u);
	EXPECT_EQ(A.size(),6u);
	f64 val = 0.0;
	A.setToZero();
	for (u32 i = 0; i < 2; i++){
		for (u32 j = 0; j < 3; j++) {
			val = A.at(i,j);
			EXPECT_EQ(val, 0.0);
		}
	}
}

TEST_F(Test, TestMatrix, copyStructure)
{
	Math::Matrix<f64> A;
	Math::Matrix<f64> B;
	B.resize(2,3);
	EXPECT_EQ(B.nRows(),2u);
	EXPECT_EQ(B.nColumns(),3u);
	EXPECT_EQ(B.size(),6u);
	A.copyStructure(B);
	EXPECT_EQ(A.nRows(),2u);
	EXPECT_EQ(A.nColumns(),3u);
	EXPECT_EQ(A.size(),6u);
	f64 val = 0.0;
	A.setToZero();
	for (u32 i = 0; i < 2; i++){
		for (u32 j = 0; j < 3; j++) {
			val = A.at(i,j);
			EXPECT_EQ(val, 0.0);
		}
	}
}

TEST_F(Test, TestMatrix, copyConstructor){
	Math::Matrix<f32> x(2,3);
	x.setToZero();
	x.at(0,0) = 1.0f;

	Math::Matrix<f32>y(x);
	EXPECT_EQ(x.nRows(), y.nRows());
	EXPECT_EQ(x.nColumns(), y.nColumns());
	EXPECT_EQ(x.at(0,0), y.at(0,0));
}

TEST_F(Test, TestMatrix, assignment)
{
	Math::Matrix<f32> x(2,3);
	x.setToZero();
	x.at(0,0) = 1.0f;

	Math::Matrix<f32>y;
	y = x;
	EXPECT_EQ(x.nRows(), y.nRows());
	EXPECT_EQ(x.nColumns(), y.nColumns());
	EXPECT_EQ(x.at(0,0), y.at(0,0));
}

TEST_F(Test, TestMatrix, copy)
{
	Math::Matrix<f32> B(1,1);
	B.copyStructure(A_);
	B.copy(A_);
	EXPECT_EQ(A_.nRows(), B.nRows());
	EXPECT_EQ(A_.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A_.at(i,j), B.at(i,j));
		}
	}
}

TEST_F(Test, TestMatrix, setToZero)
{
	Math::Matrix<f32> A(2,3);
	A.setToZero();
	for (u32 i = 0; i < A.nRows(); i++) {
		for (u32 j = 0; j < A.nColumns(); j++) {
			EXPECT_EQ(A.at(i,j), (f32)0);
		}
	}
}

TEST_F(Test, TestMatrix, binaryIO)
{
	Math::Matrix<f32> B;
	std::string filename("_tmp_testBinaryIO_tmp_.bin");
	A_.write(filename);
	B.read(filename);
	EXPECT_EQ(A_.nRows(), B.nRows());
	EXPECT_EQ(A_.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A_.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testBinaryIO_tmp_.bin");
}

TEST_F(Test, TestMatrix, gzIO)
{
	Math::Matrix<f32> B;
	std::string filename("_tmp_testGzIO_tmp_.gz");
	A_.write(filename);
	B.read(filename);
	EXPECT_EQ(A_.nRows(), B.nRows());
	EXPECT_EQ(A_.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A_.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testGzIO_tmp_.gz");
}

TEST_F(Test, TestMatrix, asciiIO)
{
	Math::Matrix<f32> B;
	std::string filename("_tmp_testAsciiIO_tmp_");
	A_.write(filename);
	B.read(filename);
	EXPECT_EQ(A_.nRows(), B.nRows());
	EXPECT_EQ(A_.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A_.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testAsciiIO_tmp_");
}

TEST_F(Test, TestMatrix, binaryIOTransposed)
{
	Math::Matrix<f32> A(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			A.at(i,j) = i * 3 + j;
		}
	}
	Math::Matrix<f32> B;
	std::string filename("_tmp_testBinaryIO_tmp_.bin");
	A.write(filename, true);
	B.read(filename, true);
	EXPECT_EQ(A.nRows(), B.nRows());
	EXPECT_EQ(A.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testBinaryIO_tmp_.bin");
}

TEST_F(Test, TestMatrix, gzIOTransposed)
{
	Math::Matrix<f32> A(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			A.at(i,j) = i * 3 + j;
		}
	}
	Math::Matrix<f32> B;
	std::string filename("_tmp_testBinaryIO_tmp_.gz");
	A.write(filename, true);
	B.read(filename, true);
	EXPECT_EQ(A.nRows(), B.nRows());
	EXPECT_EQ(A.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testBinaryIO_tmp_.gz");
}

TEST_F(Test, TestMatrix, asciiIOTransposed)
{
	Math::Matrix<f32> A(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			A.at(i,j) = i * 3 + j;
		}
	}
	Math::Matrix<f32> B;
	std::string filename("_tmp_testBinaryIO_tmp_");
	A.write(filename, true);
	B.read(filename, true);
	EXPECT_EQ(A.nRows(), B.nRows());
	EXPECT_EQ(A.nColumns(), B.nColumns());
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A.at(i,j), B.at(i,j));
		}
	}
	remove("_tmp_testBinaryIO_tmp_");
}

TEST_F(Test, TestMatrix, scale)
{
	Math::Matrix<f32> A;
	A.copyStructure(A_);
	A.copy(A_);
	A.scale(1.5);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			EXPECT_EQ(A.at(i,j), (i+j) * (f32)1.5);
		}
	}
}

TEST_F(Test, TestMatrix, add)
{
	Math::Matrix<f32> B(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			B.at(i,j) = i*j + 7;
		}
	}
	B.add(A_, (f32)1.5);
	EXPECT_EQ(B.at(0,0), (f32)7);
	EXPECT_EQ(B.at(0,1), (f32)8.5);
	EXPECT_EQ(B.at(0,2), (f32)10);
	EXPECT_EQ(B.at(1,0), (f32)8.5);
	EXPECT_EQ(B.at(1,1), (f32)11);
	EXPECT_EQ(B.at(1,2), (f32)13.5);
}

TEST_F(Test, TestMatrix, addMatrixProduct)
{
	// test four times the same computation, but with different matrices transposed

	// basic case
	Math::Matrix<f32> A(2,4);
	A.at(0,0) = 1; A.at(0,1) = -2; A.at(0,2) = -1; A.at(0,3) = 3;
	A.at(1,0) = 0; A.at(1,1) = 1; A.at(1,2) = 2; A.at(1,3) = -3;
	Math::Matrix<f32> B(4,2);
	B.at(0,0) = 0; B.at(0,1) = -2;
	B.at(1,0) = 1; B.at(1,1) = 2;
	B.at(2,0) = -2; B.at(2,1) = -3;
	B.at(3,0) = 4; B.at(3,1) = -1;
	Math::Matrix<f32> C(2,2);
	C.at(0,0) = 2; C.at(0,1) = 3;
	C.at(1,0) = 3; C.at(1,1) = -1;
	// define A^T
	Math::Matrix<f32> AT(4,2);
	AT.at(0,0) = 1; AT.at(0,1) = 0;
	AT.at(1,0) = -2; AT.at(1,1) = 1;
	AT.at(2,0) = -1; AT.at(2,1) = 2;
	AT.at(3,0) = 3; AT.at(3,1) = -3;
	// define B^T
	Math::Matrix<f32> BT(2,4);
	BT.at(0,0) = 0; BT.at(0,1) = 1; BT.at(0,2) = -2; BT.at(0,3) = 4;
	BT.at(1,0) = -2; BT.at(1,1) = 2; BT.at(1,2) = -3; BT.at(1,3) = -1;

	// C := scaleA * A * B + scaleC * C
	C.addMatrixProduct(A, B, (f32)-1.0, (f32)2.0, false, false);
	EXPECT_EQ(C.at(0,0), (f32)22);
	EXPECT_EQ(C.at(0,1), (f32)-15);
	EXPECT_EQ(C.at(1,0), (f32)-33);
	EXPECT_EQ(C.at(1,1), (f32)-1);

	// C := scaleA * AT' * B + scaleC * C
	C.at(0,0) = 2; C.at(0,1) = 3;
	C.at(1,0) = 3; C.at(1,1) = -1;
	C.addMatrixProduct(AT, B, (f32)-1.0, (f32)2.0, true, false);
	EXPECT_EQ(C.at(0,0), (f32)22);
	EXPECT_EQ(C.at(0,1), (f32)-15);
	EXPECT_EQ(C.at(1,0), (f32)-33);
	EXPECT_EQ(C.at(1,1), (f32)-1);

	// C := scaleA * A * BT' + scaleC * C
	C.at(0,0) = 2; C.at(0,1) = 3;
	C.at(1,0) = 3; C.at(1,1) = -1;
	C.addMatrixProduct(A, BT, (f32)-1.0, (f32)2.0, false, true);
	EXPECT_EQ(C.at(0,0), (f32)22);
	EXPECT_EQ(C.at(0,1), (f32)-15);
	EXPECT_EQ(C.at(1,0), (f32)-33);
	EXPECT_EQ(C.at(1,1), (f32)-1);

	// C := scaleA * AT' * BT' + scaleC * C
	C.at(0,0) = 2; C.at(0,1) = 3;
	C.at(1,0) = 3; C.at(1,1) = -1;
	C.addMatrixProduct(AT, BT, (f32)-1.0, (f32)2.0, true, true);
	EXPECT_EQ(C.at(0,0), (f32)22);
	EXPECT_EQ(C.at(0,1), (f32)-15);
	EXPECT_EQ(C.at(1,0), (f32)-33);
	EXPECT_EQ(C.at(1,1), (f32)-1);
}

TEST_F(Test, TestMatrix, elementwiseMultiplication)
{
	Math::Matrix<f32> A(2,3);
	A.at(0,0) = 10; A.at(0,1) = 9; A.at(0,2) = 4;
	A.at(1,0) = -8; A.at(1,1) = -14; A.at(1,2) = 10;
	Math::Matrix<f32> B(2,3);
	B.at(0,0) = 2; B.at(0,1) = 3; B.at(0,2) = -2;
	B.at(1,0) = -4; B.at(1,1) = 7; B.at(1,2) = -5;
	A.elementwiseMultiplication(B);
	EXPECT_EQ(A.at(0,0), (f32)20);
	EXPECT_EQ(A.at(0,1),(f32) 27);
	EXPECT_EQ(A.at(0,2), (f32)-8);
	EXPECT_EQ(A.at(1,0), (f32)32);
	EXPECT_EQ(A.at(1,1), (f32)-98);
	EXPECT_EQ(A.at(1,2), (f32)-50);
}

TEST_F(Test, TestMatrix, elementwiseDivision)
{
	Math::Matrix<f32> A(2,3);
	A.at(0,0) = 10; A.at(0,1) = 9; A.at(0,2) = 4;
	A.at(1,0) = -8; A.at(1,1) = -14; A.at(1,2) = 10;
	Math::Matrix<f32> B(2,3);
	B.at(0,0) = 2; B.at(0,1) = 3; B.at(0,2) = -2;
	B.at(1,0) = -4; B.at(1,1) = 7; B.at(1,2) = -5;
	A.elementwiseDivision(B);
	EXPECT_EQ(A.at(0,0), (f32)5);
	EXPECT_EQ(A.at(0,1), (f32)3);
	EXPECT_EQ(A.at(0,2), (f32)-2);
	EXPECT_EQ(A.at(1,0), (f32)2);
	EXPECT_EQ(A.at(1,1), (f32)-2);
	EXPECT_EQ(A.at(1,2), (f32)-2);
}
