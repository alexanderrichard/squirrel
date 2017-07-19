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

class TestVector : public Test::Fixture
{
protected:
	Math::Vector<f32> v_;
public:
	void setUp();
	void tearDown();
};

void TestVector::setUp()
{
	v_.resize(3);
	for (u32 i = 0; i < 3; i++) {
		v_.at(i) = i;
	}
}

void TestVector::tearDown()
{
}

TEST_F(Test, TestVector, binaryIO)
{
	Math::Vector<f32> w;
	std::string filename("_tmp_testBinaryIO_tmp_.bin");
	v_.write(filename);
	w.read(filename);
	EXPECT_EQ(v_.nRows(), w.nRows());
	for (u32 i = 0; i < 3; i++) {
		EXPECT_EQ(v_.at(i), w.at(i));
	}
	remove("_tmp_testBinaryIO_tmp_.bin");
}

TEST_F(Test, TestVector, asciiIO)
{
	Math::Vector<f32> w;
	std::string filename("_tmp_testAsciiIO_tmp_");
	v_.write(filename);
	w.read(filename);
	EXPECT_EQ(v_.nRows(), w.nRows());
	for (u32 i = 0; i < 3; i++) {
		EXPECT_EQ(v_.at(i), w.at(i));
	}
	remove("_tmp_testAsciiIO_tmp_");
}

TEST_F(Test, TestVector, dot)
{
	Math::Vector<f32> w(3);
	w.at(0) = 3;
	w.at(1) = 2;
	w.at(2) = 1;
	EXPECT_EQ(v_.dot(w), (f32)4);
}

TEST_F(Test, TestVector, chiSquareDistance)
{
	Math::Vector<f32> v(3);
	v.at(0) = 1;
	v.at(1) = 2;
	v.at(2) = 3;
	Math::Vector<f32> w(3);
	w.at(0) = 3;
	w.at(1) = 2;
	w.at(2) = 1;
	f32 d = w.chiSquareDistance(v);
	EXPECT_EQ(d, (f32)1);
}

TEST_F(Test, TestVector, add)
{
	Math::Vector<f32> w(3);
	w.at(0) = 3;
	w.at(1) = 2;
	w.at(2) = 1;
	w.add(v_, (f32)1.5);
	EXPECT_EQ(w.at(0), (f32)3);
	EXPECT_EQ(w.at(1), (f32)3.5);
	EXPECT_EQ(w.at(2), (f32)4);
}

TEST_F(Test, TestVector, multiply)
{
	Math::Matrix<f32> A(2,3);
	for (u32 i = 0; i < 2; i++) {
		for (u32 j = 0; j < 3; j++) {
			A.at(i,j) = i+j;
		}
	}
	Math::Vector<f32> w(2);
	w.at(0) = 10;
	w.at(1) = 5;
	w.multiply(A, v_, false, 1.5, -2.0);
	EXPECT_EQ(w.at(0), (f32)-12.5);
	EXPECT_EQ(w.at(1), (f32)2);

	w.at(0) = 2;
	w.at(1) = 4;
	Math::Vector<f32> x(3);
	x.at(0) = 1;
	x.at(1) = 2;
	x.at(2) = -1;
	x.multiply(A, w, true, -1.0, 3.0);
	EXPECT_EQ(x.at(0), (f32)-1);
	EXPECT_EQ(x.at(1), (f32)-4);
	EXPECT_EQ(x.at(2), (f32)-19);
}

TEST_F(Test, TestVector, elementwiseMultiplication)
{
	Math::Vector<f32> v(3);
	v.at(0) = 0; v.at(1) = -2; v.at(2) = 2;
	Math::Vector<f32> w(3);
	w.at(0) = -3; w.at(1) = 4; w.at(2) = -10;
	v.elementwiseMultiplication(w);
	EXPECT_EQ(v.at(0), (f32)0);
	EXPECT_EQ(v.at(1), (f32)-8);
	EXPECT_EQ(v.at(2), (f32)-20);
}
