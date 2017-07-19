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
 * Core_HashMap.cc
 *
 *  Created on: Jun 16, 2017
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <Core/HashMap.hh>

using namespace std;

class TestHashMap : public Test::Fixture
{
public:
	static u32 hash_fctn(const u32& n) { return n; }
	void setUp();
	void tearDown();
};

void TestHashMap::setUp() {
}

void TestHashMap::tearDown() {
}

TEST_F(Test, TestHashMap, insert) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	Core::HashMap<u32, Float>::Entry *entry = map.insert(10, 0.1f);
	EXPECT_EQ(entry->key, 10u);
	EXPECT_EQ(entry->value, 0.1f);
	entry = map.insert(20, 0.2f);
	EXPECT_EQ(entry->key, 20u);
	EXPECT_EQ(entry->value, 0.2f);
	entry = map.insert(5, 0.05f);
	EXPECT_EQ(entry->key, 5u);
	EXPECT_EQ(entry->value, 0.05f);
}

TEST_F(Test, TestHashMap, begin) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	Core::HashMap<u32, Float>::Entry* ptr = map.begin();
	EXPECT_EQ(ptr, (Core::HashMap<u32, Float>::Entry*)0);
	map.insert(10, 0.1);
	ptr = map.begin();
	EXPECT_GT(ptr, (Core::HashMap<u32, Float>::Entry*)0);
	map.remove(10);
	ptr = map.begin();
	EXPECT_EQ(ptr, (Core::HashMap<u32, Float>::Entry*)0);
}

TEST_F(Test, TestHashMap, size) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1);
	map.insert(20, 0.2);
	map.insert(5, 0.05);
	map.insert(3, 0.03);
	map.insert(2, 0.02);
	EXPECT_EQ(map.size(), 5u);
}

TEST_F(Test, TestHashMap, iterate) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1f);
	map.insert(20, 0.2f);
	map.insert(5, 0.05f);
	map.insert(3, 0.03f);
	map.insert(2, 0.02f);
	std::vector<bool> found(5, false);
	for (Core::HashMap<u32, Float>::Entry* it = map.begin(); it != map.end(); it = it->next) {
		if ((it->key == 10) && (it->value == 0.1f)) found.at(0) = true;
		if ((it->key == 20) && (it->value == 0.2f)) found.at(1) = true;
		if ((it->key == 5) && (it->value == 0.05f)) found.at(2) = true;
		if ((it->key == 3) && (it->value == 0.03f)) found.at(3) = true;
		if ((it->key == 2) && (it->value == 0.02f)) found.at(4) = true;
	}
	for (u32 i = 0; i < 5; i++) {
		EXPECT_TRUE(found.at(i));
	}
}

TEST_F(Test, TestHashMap, insertExisting) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1f);
	map.insert(20, 0.2f);
	map.insert(5, 0.05f);
	map.insert(5, 5.0f);

	EXPECT_EQ(map.size(), 3u);

	std::vector<bool> found(3, false);
	for (Core::HashMap<u32, Float>::Entry* it = map.begin(); it != map.end(); it = it->next) {
		if ((it->key == 10) && (it->value == 0.1f)) found.at(0) = true;
		if ((it->key == 20) && (it->value == 0.2f)) found.at(1) = true;
		if ((it->key == 5) && (it->value == 5.0f)) found.at(2) = true;
	}
	for (u32 i = 0; i < 3; i++) {
		EXPECT_TRUE(found.at(i));
	}
}

TEST_F(Test, TestHashMap, remove) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1f);
	map.insert(20, 0.2f);
	map.insert(5, 0.05f);
	map.insert(3, 0.03f);
	map.insert(2, 0.02f);

	map.remove(100);
	EXPECT_EQ(map.size(), 5u);
	map.remove(10);
	EXPECT_EQ(map.size(), 4u);
	map.remove(5);
	EXPECT_EQ(map.size(), 3u);
	map.remove(2);
	EXPECT_EQ(map.size(), 2u);
	std::vector<bool> found(2, false);
	for (Core::HashMap<u32, Float>::Entry* it = map.begin(); it != map.end(); it = it->next) {
		if ((it->key == 20) && (it->value == 0.2f)) found.at(0) = true;
		if ((it->key == 3) && (it->value == 0.03f)) found.at(1) = true;
	}
	EXPECT_TRUE(found.at(0));
	EXPECT_TRUE(found.at(1));
}

TEST_F(Test, TestHashMap, find) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1f);
	map.insert(20, 0.2f);
	map.insert(5, 0.05f);

	Core::HashMap<u32, Float>::Entry* entry = map.find(20);
	EXPECT_GT(entry, (Core::HashMap<u32, Float>::Entry*)0);
	EXPECT_EQ(entry->key, 20u);
	EXPECT_EQ(entry->value, 0.2f);

	entry = map.find(50);
	EXPECT_EQ(entry, (Core::HashMap<u32, Float>::Entry*)0);
}

TEST_F(Test, TestHashMap, clear) {
	Core::HashMap<u32, Float> map(hash_fctn, 11);
	map.insert(10, 0.1f);
	map.insert(20, 0.2f);
	map.insert(5, 0.05f);

	map.clear();
	EXPECT_EQ(map.size(), 0u);
	EXPECT_EQ(map.begin(), (Core::HashMap<u32, Float>::Entry*)0);
}
