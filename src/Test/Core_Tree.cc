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
 * Core_Tree.cc
 *
 *  Created on: Jul 1, 2015
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <Core/Tree.hh>

using namespace std;

class TestTree : public Test::Fixture
{
public:
	void addPaths(Core::Tree<u32, Float>& tree);
	void setUp();
	void tearDown();
};

void TestTree::addPaths(Core::Tree<u32, Float>& tree) {
	Core::Tree<u32, Float>::Path path(1, 0);
	tree.addNode(path, 1, 1.0);
	path.push_back(1);
	tree.addNode(path, 11, 1.1);
	path.push_back(11);
	tree.addNode(path, 111, 1.11);
	tree.addNode(path, 112, 1.12);
	path.pop_back();
	tree.addNode(path, 12, 1.2);
	path.push_back(12);
	tree.addNode(path, 121, 1.21);
	path.pop_back();
	path.pop_back();
	tree.addNode(path, 2, 2.0);
	tree.addNode(path, 3, 3.0);
	path.push_back(3);
	tree.addNode(path, 31, 3.1);
	tree.addNode(path, 32, 3.2);
}

void TestTree::setUp() {
}

void TestTree::tearDown() {
}

TEST_F(Test, TestTree, addNode) {
	Core::Tree<u32, Float> tree(0, 0.0);
	Core::Tree<u32, Float>::Path p;
	p.push_back(0);
	EXPECT_TRUE(tree.addNode(p, 1, 1.0));
	EXPECT_FALSE(tree.addNode(p, 1, 1.0));
	p.push_back(2);
	EXPECT_FALSE(tree.addNode(p, 1, 1.0));
}

TEST_F(Test, TestTree, keyExists) {
	Core::Tree<u32, Float> tree(0, 0.0);
	addPaths(tree);
	EXPECT_TRUE(tree.keyExists(112));
	EXPECT_FALSE(tree.keyExists(122));
}

TEST_F(Test, TestTree, pathExists) {
	Core::Tree<u32, Float> tree(0, 0.0);
	addPaths(tree);
	Core::Tree<u32, Float>::Path p;
	p.push_back(0);
	p.push_back(1);
	p.push_back(12);
	EXPECT_TRUE(tree.pathExists(p));
	p.push_back(122);
	EXPECT_FALSE(tree.pathExists(p));
}

TEST_F(Test, TestTree, setValue) {
	Core::Tree<u32, Float> tree(0, 0.0);
	addPaths(tree);
	Core::Tree<u32, Float>::Path p;
	p.push_back(0);
	p.push_back(1);
	p.push_back(12);
	EXPECT_TRUE(tree.setValue(p, 0.5));
	EXPECT_EQ(tree.value(p), 0.5f);
	p.push_back(122);
	EXPECT_FALSE(tree.setValue(p, 0.7));
}

TEST_F(Test, TestTree, value) {
	Core::Tree<u32, Float> tree(0, 0.0);
	addPaths(tree);
	Core::Tree<u32, Float>::Path p;
	p.push_back(0);
	p.push_back(1);
	p.push_back(12);
	tree.value(p) = 0.5;
	EXPECT_EQ(tree.value(p), 0.5f);
	p.pop_back();
	EXPECT_EQ(tree.value(p), 1.0f);
}

TEST_F(Test, TestTree, addPath) {
	Core::Tree<u32, Float> tree(0, 0.0);
	addPaths(tree);

	// add a valid path
	Core::Tree<u32, Float>::Path p;
	p.push_back(0);
	p.push_back(2);
	p.push_back(21);
	p.push_back(211);
	EXPECT_TRUE(tree.addPath(p, 0.5));
	EXPECT_EQ(tree.value(p), 0.5f);
	p.pop_back();
	EXPECT_EQ(tree.value(p), 0.5f);
	p.pop_back();
	EXPECT_EQ(tree.value(p), 2.0f);
	p.pop_back();
	EXPECT_EQ(tree.value(p), 0.0f);

	// add an invalid path
	p.clear();
	p.push_back(1);
	p.push_back(2);
	EXPECT_FALSE(tree.addPath(p, 0.5));

	// add an existing path
	p.clear();
	p.push_back(0);
	p.push_back(1);
	EXPECT_TRUE(tree.addPath(p, 0.5));
	EXPECT_EQ(tree.value(p), 1.0f);
	p.pop_back();
	EXPECT_EQ(tree.value(p), 0.0f);
}
