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
#include <Features/Preprocessor.hh>
#include <Math/Matrix.hh>
#include <Math/Vector.hh>

class TestPreprocessor : public Test::Fixture
{
protected:
	class TestVectorSubtraction : public Features::VectorSubtractionPreprocessor {
		friend class TestPreprocessor;
	public:
		TestVectorSubtraction() : VectorSubtractionPreprocessor("vector-subtraction-preprocessor") {}
	};
	class TestVectorDivision : public Features::VectorDivisionPreprocessor {
		friend class TestPreprocessor;
	public:
		TestVectorDivision() : VectorDivisionPreprocessor("vector-division-preprocessor") {}
	};
	class TestMatrixMultiplication : public Features::MatrixMultiplicationPreprocessor {
		friend class TestPreprocessor;
	public:
		TestMatrixMultiplication() : MatrixMultiplicationPreprocessor("matrix-multiplication-preprocessor") {}
	};
	class TestFeatureSelection : public Features::FeatureSelectionPreprocessor {
		friend class TestPreprocessor;
	public:
		TestFeatureSelection() : FeatureSelectionPreprocessor("feature-selection-preprocessor") {}
	};
	class TestWindowing : public Features::WindowingPreprocessor {
		friend class TestPreprocessor;
	public:
		TestWindowing() : WindowingPreprocessor("feature-selection-preprocessor") {}
	};
	TestVectorSubtraction* vecSub_;
	TestVectorDivision* vecDiv_;
	TestMatrixMultiplication* matMul_;
	TestFeatureSelection* feaSel_;
	TestWindowing* window_;
public:
	void setUp();
	void tearDown();
};

void TestPreprocessor::setUp()
{
	Core::Configuration::initialize();
	/* set up vector-subtraction preprocessor */
	vecSub_ = new TestVectorSubtraction;
	vecSub_->isInitialized_ = true;
	vecSub_->inputDimension_ = 2;
	vecSub_->outputDimension_ = 2;
	vecSub_->vector_.resize(2);
	vecSub_->vector_.at(0) = 2;
	vecSub_->vector_.at(1) = -1;
	/* set up vector-division preprocessor */
	vecDiv_ = new TestVectorDivision;
	vecDiv_->isInitialized_ = true;
	vecDiv_->inputDimension_ = 2;
	vecDiv_->outputDimension_ = 2;
	vecDiv_->vector_.resize(2);
	vecDiv_->vector_.at(0) = 2;
	vecDiv_->vector_.at(1) = -1;
	/* set up matrix-multiplication preprocessor */
	matMul_ = new TestMatrixMultiplication;
	matMul_->isInitialized_ = true;
	matMul_->inputDimension_ = 2;
	matMul_->outputDimension_ = 1;
	matMul_->matrix_.resize(1,2);
	matMul_->matrix_.at(0,0) = 2;
	matMul_->matrix_.at(0,1) = 1;
	/* set up feature-selection preprocessor */
	feaSel_ = new TestFeatureSelection;
	feaSel_->startIndex_ = 1;
	feaSel_->endIndex_ = 2;
	feaSel_->initialize(4);
	/* set up windowing preprocessor */
	window_ = new TestWindowing;
	window_->windowSize_ = 3;
	window_->initialize(2);
}

void TestPreprocessor::tearDown()
{
	delete vecSub_;
	delete vecDiv_;
	delete matMul_;
	delete feaSel_;
	delete window_;
}

TEST_F(Test, TestPreprocessor, VectorSubtractionPreprocessor)
{
	Math::Matrix<Float> m_in(2,2);
	m_in.at(0,0) = -2;
	m_in.at(1,0) = 2;
	m_in.at(0,1) = 3;
	m_in.at(1,1) = -1;
	Math::Matrix<Float> m_out;
	vecSub_->work(m_in, m_out);
	EXPECT_EQ((Float)-4, m_out.at(0,0));
	EXPECT_EQ((Float)3, m_out.at(1,0));
	EXPECT_EQ((Float)1, m_out.at(0,1));
	EXPECT_EQ((Float)0, m_out.at(1,1));
}

TEST_F(Test, TestPreprocessor, VectorDivisionPreprocessor)
{
	Math::Matrix<Float> m_in(2,2);
	m_in.at(0,0) = -2;
	m_in.at(1,0) = 2;
	m_in.at(0,1) = 3;
	m_in.at(1,1) = -1;
	Math::Matrix<Float> m_out;
	vecDiv_->work(m_in, m_out);
	EXPECT_EQ((Float)-1, m_out.at(0,0));
	EXPECT_EQ((Float)-2, m_out.at(1,0));
	EXPECT_EQ((Float)1.5, m_out.at(0,1));
	EXPECT_EQ((Float)1, m_out.at(1,1));
}

TEST_F(Test, TestPreprocessor, MatrixMultiplicationPreprocessor)
{
	Math::Matrix<Float> m_in(2,2);
	m_in.at(0,0) = -2;
	m_in.at(1,0) = 2;
	m_in.at(0,1) = 3;
	m_in.at(1,1) = -1;
	Math::Matrix<Float> m_out;
	matMul_->work(m_in, m_out);
	EXPECT_EQ((Float)-2, m_out.at(0,0));
	EXPECT_EQ((Float)5, m_out.at(0,1));
}

TEST_F(Test, TestPreprocessor, FeatureSelectionPreprocessor)
{
	Math::Matrix<Float> m_in(4,2);
	m_in.at(0,0) = 0;
	m_in.at(1,0) = 1;
	m_in.at(2,0) = 2;
	m_in.at(3,0) = 3;
	m_in.at(0,1) = 4;
	m_in.at(1,1) = 5;
	m_in.at(2,1) = 6;
	m_in.at(3,1) = 7;
	Math::Matrix<Float> m_out;
	feaSel_->work(m_in, m_out);
	EXPECT_EQ((Float)1, m_out.at(0,0));
	EXPECT_EQ((Float)2, m_out.at(1,0));
	EXPECT_EQ((Float)5, m_out.at(0,1));
	EXPECT_EQ((Float)6, m_out.at(1,1));
}

TEST_F(Test, TestPreprocessor, WindowingPreprocessor)
{
	Math::Matrix<Float> m_in(2,4);
	m_in.at(0,0) = 1;
	m_in.at(1,0) = 2;
	m_in.at(0,1) = 3;
	m_in.at(1,1) = 4;
	m_in.at(0,2) = 5;
	m_in.at(1,2) = 6;
	m_in.at(0,3) = 7;
	m_in.at(1,3) = 8;
	Math::Matrix<Float> m_out;
	window_->work(m_in, m_out);
	EXPECT_EQ(6u, m_out.nRows());
	EXPECT_EQ(4u, m_out.nColumns());
	// first column
	EXPECT_EQ((Float)0, m_out.at(0,0));
	EXPECT_EQ((Float)0, m_out.at(1,0));
	EXPECT_EQ((Float)1, m_out.at(2,0));
	EXPECT_EQ((Float)2, m_out.at(3,0));
	EXPECT_EQ((Float)3, m_out.at(4,0));
	EXPECT_EQ((Float)4, m_out.at(5,0));
	// second column
	EXPECT_EQ((Float)1, m_out.at(0,1));
	EXPECT_EQ((Float)2, m_out.at(1,1));
	EXPECT_EQ((Float)3, m_out.at(2,1));
	EXPECT_EQ((Float)4, m_out.at(3,1));
	EXPECT_EQ((Float)5, m_out.at(4,1));
	EXPECT_EQ((Float)6, m_out.at(5,1));
	// third column
	EXPECT_EQ((Float)3, m_out.at(0,2));
	EXPECT_EQ((Float)4, m_out.at(1,2));
	EXPECT_EQ((Float)5, m_out.at(2,2));
	EXPECT_EQ((Float)6, m_out.at(3,2));
	EXPECT_EQ((Float)7, m_out.at(4,2));
	EXPECT_EQ((Float)8, m_out.at(5,2));
	// fourth column
	EXPECT_EQ((Float)5, m_out.at(0,3));
	EXPECT_EQ((Float)6, m_out.at(1,3));
	EXPECT_EQ((Float)7, m_out.at(2,3));
	EXPECT_EQ((Float)8, m_out.at(3,3));
	EXPECT_EQ((Float)0, m_out.at(4,3));
	EXPECT_EQ((Float)0, m_out.at(5,3));
}
