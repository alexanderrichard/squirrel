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
#include <Lm/LanguageModel.hh>

using namespace std;

class TestLanguageModel : public Test::Fixture
{
public:
	std::vector<Lm::Word> sequenceA_;
	std::vector<Lm::Word> sequenceB_;
	void setUp();
	void tearDown();
};

void TestLanguageModel::setUp() {
	Core::Configuration::setParameter("language-model.lexicon-size", "3");
	Lm::Word s1[] = {0, 1, 1, 0, 2, 1, 0};
	Lm::Word s2[] = {2, 2, 1, 2, 0, 1, 2};
	sequenceA_ = std::vector<Lm::Word>(s1, s1 + sizeof(s1) / sizeof(s1[0]));
	sequenceB_ = std::vector<Lm::Word>(s2, s2 + sizeof(s2) / sizeof(s2[0]));
}

void TestLanguageModel::tearDown() {
	Core::Configuration::reset();
}

TEST_F(Test, TestLanguageModel, zerogram) {
	Core::Configuration::setParameter("language-model.type", "zerogram");
	Lm::NGram lm;
	EXPECT_EQ(1.0f / 3.0f, lm.probability());
}

TEST_F(Test, TestLanguageModel, unigram) {
	Core::Configuration::setParameter("language-model.type", "unigram");
	Core::Configuration::setParameter("language-model.backing-off", "true");
	Lm::NGram lm;
	lm.accumulate(sequenceA_);
	lm.accumulate(sequenceB_);
	lm.estimateDiscountingParameter();
	EXPECT_EQ(4.0f / 14.0f, lm.probability(0));
	EXPECT_EQ(5.0f / 14.0f, lm.probability(1));
	EXPECT_EQ(5.0f / 14.0f, lm.probability(2));
}

TEST_F(Test, TestLanguageModel, bigramBackingOff) {
	Core::Configuration::setParameter("language-model.type", "bigram");
	Core::Configuration::setParameter("language-model.backing-off", "true");
	Lm::NGram lm;
	lm.accumulate(sequenceA_);
	lm.accumulate(sequenceB_);
	lm.estimateDiscountingParameter();

	EXPECT_DOUBLE_EQ(2.0f / 7.0f, lm.probability(0, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(3.0f / 7.0f, lm.probability(1, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 7.0f, lm.probability(2, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(3.0f / 7.0f, lm.probability(0, 0), 1e-5);
	EXPECT_DOUBLE_EQ(8.0f / 21.0f, lm.probability(1, 0), 1e-5);
	EXPECT_DOUBLE_EQ(4.0f / 21.0f, lm.probability(2, 0), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 5.0f, lm.probability(0, 1), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 5.0f, lm.probability(1, 1), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 5.0f, lm.probability(2, 1), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 4.0f, lm.probability(0, 2), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 4.0f, lm.probability(1, 2), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 4.0f, lm.probability(2, 2), 1e-5);
}

TEST_F(Test, TestLanguageModel, loadSaveLm) {
	Core::Configuration::setParameter("language-model.type", "bigram");
	Core::Configuration::setParameter("language-model.backing-off", "true");
	Core::Configuration::setParameter("language-model.file", "_tmp_testLoadSaveLm_tmp_.lm");
	Lm::NGram lm;
	lm.accumulate(sequenceA_);
	lm.accumulate(sequenceB_);
	lm.estimateDiscountingParameter();
	lm.saveModel();
	Lm::NGram lm2;
	lm2.loadModel();
	EXPECT_DOUBLE_EQ(2.0f / 7.0f, lm2.probability(0, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(3.0f / 7.0f, lm2.probability(1, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 7.0f, lm2.probability(2, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(3.0f / 7.0f, lm2.probability(0, 0), 1e-5);
	EXPECT_DOUBLE_EQ(8.0f / 21.0f, lm2.probability(1, 0), 1e-5);
	EXPECT_DOUBLE_EQ(4.0f / 21.0f, lm2.probability(2, 0), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 5.0f, lm2.probability(0, 1), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 5.0f, lm2.probability(1, 1), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 5.0f, lm2.probability(2, 1), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 4.0f, lm2.probability(0, 2), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 4.0f, lm2.probability(1, 2), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 4.0f, lm2.probability(2, 2), 1e-5);
	remove("_tmp_testLoadSaveLm_tmp_.lm");
}

TEST_F(Test, TestLanguageModel, bigramNoBackingOff) {
	Core::Configuration::setParameter("language-model.type", "bigram");
	Core::Configuration::setParameter("language-model.backing-off", "false");
	Lm::NGram lm;
	lm.accumulate(sequenceA_);
	lm.accumulate(sequenceB_);
	lm.estimateDiscountingParameter();
	EXPECT_DOUBLE_EQ(0.5f, lm.probability(0, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(0.0f, lm.probability(1, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(0.5f, lm.probability(2, Lm::NGram::senStart), 1e-5);
	EXPECT_DOUBLE_EQ(0.0f, lm.probability(0, 0), 1e-5);
	EXPECT_DOUBLE_EQ(2.0f / 3.0f, lm.probability(1, 0), 1e-5);
	EXPECT_DOUBLE_EQ(1.0f / 3.0f, lm.probability(2, 0), 1e-5);
	EXPECT_DOUBLE_EQ(0.4f, lm.probability(0, 1), 1e-5);
	EXPECT_DOUBLE_EQ(0.2f, lm.probability(1, 1), 1e-5);
	EXPECT_DOUBLE_EQ(0.4f, lm.probability(2, 1), 1e-5);
	EXPECT_DOUBLE_EQ(0.25f, lm.probability(0, 2), 1e-5);
	EXPECT_DOUBLE_EQ(0.5f, lm.probability(1, 2), 1e-5);
	EXPECT_DOUBLE_EQ(0.25f, lm.probability(2, 2), 1e-5);
}
