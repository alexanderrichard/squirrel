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
 * Nn_MatrixContainer.cc
 *
 *  Created on: Sep 28, 2016
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <Nn/Types.hh>
#include <Nn/MinibatchGenerator.hh>

using namespace std;

class TestMatrixContainer : public Test::Fixture
{
public:
	void setUp() {}
	void tearDown() {}
};

TEST_F(Test, TestMatrixContainer, copy) {

	Nn::MatrixContainer containerA;
	containerA.initComputation();
	containerA.setMaximalMemory(4);
	containerA.addTimeframe(2, 1);
	containerA.addTimeframe(2, 1);
	containerA.addTimeframe(2, 3);
	containerA.addTimeframe(2, 3);
	// fill the matrices in containerA
	containerA.finishComputation();
	containerA.at(0).at(0, 0) = 1.0;
	containerA.at(0).at(1, 0) = 2.0;
	containerA.at(1).at(0, 0) = 3.0;
	containerA.at(1).at(1, 0) = 4.0;
	containerA.at(2).at(0, 0) = 5.0;
	containerA.at(2).at(1, 0) = 6.0;
	containerA.at(2).at(0, 1) = 7.0;
	containerA.at(2).at(1, 1) = 8.0;
	containerA.at(2).at(0, 2) = 9.0;
	containerA.at(2).at(1, 2) = 10.0;
	containerA.at(3).at(0, 0) = 11.0;
	containerA.at(3).at(1, 0) = 12.0;
	containerA.at(3).at(0, 1) = 13.0;
	containerA.at(3).at(1, 1) = 14.0;
	containerA.at(3).at(0, 2) = 15.0;
	containerA.at(3).at(1, 2) = 16.0;
	containerA.initComputation();

	Nn::MatrixContainer containerB;
	containerB.initComputation();
	containerB.copy(containerA);
	containerB.finishComputation();

	EXPECT_EQ(containerB.at(0).at(0,0), 1.0f);
	EXPECT_EQ(containerB.at(0).at(1,0), 2.0f);
	EXPECT_EQ(containerB.at(1).at(0,0), 3.0f);
	EXPECT_EQ(containerB.at(1).at(1,0), 4.0f);
	EXPECT_EQ(containerB.at(2).at(0,0), 5.0f);
	EXPECT_EQ(containerB.at(2).at(1,0), 6.0f);
	EXPECT_EQ(containerB.at(2).at(0,1), 7.0f);
	EXPECT_EQ(containerB.at(2).at(1,1), 8.0f);
	EXPECT_EQ(containerB.at(2).at(0,2), 9.0f);
	EXPECT_EQ(containerB.at(2).at(1,2), 10.0f);
	EXPECT_EQ(containerB.at(3).at(0,0), 11.0f);
	EXPECT_EQ(containerB.at(3).at(1,0), 12.0f);
	EXPECT_EQ(containerB.at(3).at(0,1), 13.0f);
	EXPECT_EQ(containerB.at(3).at(1,1), 14.0f);
	EXPECT_EQ(containerB.at(3).at(0,2), 15.0f);
	EXPECT_EQ(containerB.at(3).at(1,2), 16.0f);
}

TEST_F(Test, TestMatrixContainer, revertTemporalOrder) {

	Nn::MatrixContainer containerA;
	containerA.initComputation();
	containerA.setMaximalMemory(4);
	containerA.addTimeframe(2, 1);
	containerA.addTimeframe(2, 1);
	containerA.addTimeframe(2, 3);
	containerA.addTimeframe(2, 3);
	// fill the matrices in containerA
	containerA.finishComputation();
	containerA.at(0).at(0, 0) = 1.0;
	containerA.at(0).at(1, 0) = 2.0;
	containerA.at(1).at(0, 0) = 3.0;
	containerA.at(1).at(1, 0) = 4.0;
	containerA.at(2).at(0, 0) = 5.0;
	containerA.at(2).at(1, 0) = 6.0;
	containerA.at(2).at(0, 1) = 7.0;
	containerA.at(2).at(1, 1) = 8.0;
	containerA.at(2).at(0, 2) = 9.0;
	containerA.at(2).at(1, 2) = 10.0;
	containerA.at(3).at(0, 0) = 11.0;
	containerA.at(3).at(1, 0) = 12.0;
	containerA.at(3).at(0, 1) = 13.0;
	containerA.at(3).at(1, 1) = 14.0;
	containerA.at(3).at(0, 2) = 15.0;
	containerA.at(3).at(1, 2) = 16.0;
	containerA.initComputation();

	containerA.revertTemporalOrder();
	containerA.finishComputation();

	EXPECT_EQ(containerA.at(0).at(0,0), 11.0f);
	EXPECT_EQ(containerA.at(0).at(1,0), 12.0f);
	EXPECT_EQ(containerA.at(1).at(0,0), 5.0f);
	EXPECT_EQ(containerA.at(1).at(1,0), 6.0f);
	EXPECT_EQ(containerA.at(2).at(0,0), 3.0f);
	EXPECT_EQ(containerA.at(2).at(1,0), 4.0f);
	EXPECT_EQ(containerA.at(2).at(0,1), 13.0f);
	EXPECT_EQ(containerA.at(2).at(1,1), 14.0f);
	EXPECT_EQ(containerA.at(2).at(0,2), 15.0f);
	EXPECT_EQ(containerA.at(2).at(1,2), 16.0f);
	EXPECT_EQ(containerA.at(3).at(0,0), 1.0f);
	EXPECT_EQ(containerA.at(3).at(1,0), 2.0f);
	EXPECT_EQ(containerA.at(3).at(0,1), 7.0f);
	EXPECT_EQ(containerA.at(3).at(1,1), 8.0f);
	EXPECT_EQ(containerA.at(3).at(0,2), 9.0f);
	EXPECT_EQ(containerA.at(3).at(1,2), 10.0f);
}
