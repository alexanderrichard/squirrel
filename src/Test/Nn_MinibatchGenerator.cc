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
 * Nn_MinibatchGenerator.cc
 *
 *  Created on: Jun 6, 2016
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <Nn/Types.hh>
#include <Nn/MinibatchGenerator.hh>

using namespace std;

class TestMinibatchGenerator : public Test::Fixture
{
public:
	void setUp() {}
	void tearDown() {}
};

TEST_F(Test, TestMinibatchGenerator, unsupervisedSingle) {

	Core::Configuration::setParameter("features.feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("source-type", "single");

	Nn::MinibatchGenerator generator(Nn::unsupervised);
	generator.initialize();
	u32 dataCount = 0;
	for (u32 batch = 0; batch < 3; batch++) {
		generator.generateBatch(4);
		EXPECT_EQ(4u, generator.sourceBatch().nColumns());
		EXPECT_EQ(3u, generator.sourceBatch().nRows());
		for (u32 col = 0; col < generator.sourceBatch().nColumns(); col++) {
			for (u32 d = 0; d < 3; d++) {
				EXPECT_EQ((dataCount + col) * 3.0f + d, generator.sourceBatch().at(d, col));
			}
		}
		dataCount += 4;
	}
	EXPECT_EQ(12u, dataCount);

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedClassificationSingleToSingle) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("features.aligned-feature-reader.label-cache", "labels-1.vectors");
	Core::Configuration::setParameter("source-type", "single");
	Core::Configuration::setParameter("target-type", "single");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();
	u32 dataCount = 0;
	for (u32 batch = 0; batch < 3; batch++) {
		generator.generateBatch(4);
		EXPECT_EQ(4u, generator.sourceBatch().nColumns());
		EXPECT_EQ(3u, generator.sourceBatch().nRows());
		EXPECT_EQ(4u, generator.targetBatch().nColumns());
		EXPECT_EQ(12u, generator.targetBatch().nRows());
		for (u32 col = 0; col < generator.sourceBatch().nColumns(); col++) {
			for (u32 d = 0; d < 3; d++) {
				EXPECT_EQ((dataCount + col) * 3.0f + d, generator.sourceBatch().at(d, col));
			}
		}
		for (u32 col = 0; col < generator.targetBatch().nColumns(); col++) {
			for (u32 d = 0; d < 12; d++) {
				Float expected = (dataCount + col == d ? 1.0 : 0.0);
				EXPECT_EQ(expected, generator.targetBatch().at(d, col));
			}
		}
		dataCount += 4;
	}
	EXPECT_EQ(12u, dataCount);

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedRegressionSingleToSingle) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("features.aligned-feature-reader.target-cache", "targets-1.vectors");
	Core::Configuration::setParameter("source-type", "single");
	Core::Configuration::setParameter("target-type", "single");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();
	u32 dataCount = 0;
	for (u32 batch = 0; batch < 3; batch++) {
		generator.generateBatch(4);
		EXPECT_EQ(4u, generator.sourceBatch().nColumns());
		EXPECT_EQ(3u, generator.sourceBatch().nRows());
		EXPECT_EQ(4u, generator.targetBatch().nColumns());
		EXPECT_EQ(2u, generator.targetBatch().nRows());
		for (u32 col = 0; col < generator.sourceBatch().nColumns(); col++) {
			for (u32 d = 0; d < 3; d++) {
				EXPECT_EQ((dataCount + col) * 3.0f + d, generator.sourceBatch().at(d, col));
			}
		}
		for (u32 col = 0; col < generator.targetBatch().nColumns(); col++) {
			for (u32 d = 0; d < 2; d++) {
				EXPECT_EQ((dataCount + col) * 2.0f + d, generator.targetBatch().at(d, col));
			}
		}
		dataCount += 4;
	}
	EXPECT_EQ(12u, dataCount);

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, unsupervisedSequence) {

	Core::Configuration::setParameter("features.feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("source-type", "sequence");

	Nn::MinibatchGenerator generator(Nn::unsupervised);
	generator.initialize();

	generator.generateBatch(2);
	EXPECT_EQ(4u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(9.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}

	generator.generateBatch(2);
	EXPECT_EQ(5u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(21.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(24.0f + d, generator.sourceSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedClassifiactionSequenceToSingle) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("features.aligned-feature-reader.label-cache", "labels-2.vectors");
	Core::Configuration::setParameter("source-type", "sequence");
	Core::Configuration::setParameter("target-type", "single");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();

	generator.generateBatch(2);
	EXPECT_EQ(4u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(9.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ((d == 1 ? 1.0f : 0.0f), generator.targetBatch().at(d, 0));
		EXPECT_EQ((d == 0 ? 1.0f : 0.0f), generator.targetBatch().at(d, 1));
	}

	generator.generateBatch(2);
	EXPECT_EQ(5u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(21.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(24.0f + d, generator.sourceSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ((d == 2 ? 1.0f : 0.0f), generator.targetBatch().at(d, 0));
		EXPECT_EQ((d == 0 ? 1.0f : 0.0f), generator.targetBatch().at(d, 1));
	}

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedRegressionSequenceToSingle) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("features.aligned-feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("source-type", "sequence");
	Core::Configuration::setParameter("target-type", "single");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();

	generator.generateBatch(2);
	EXPECT_EQ(4u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(9.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 2; d++) {
		EXPECT_EQ(2.0f + d, generator.targetBatch().at(d, 0));
		EXPECT_EQ(0.0f + d, generator.targetBatch().at(d, 1));
	}

	generator.generateBatch(2);
	EXPECT_EQ(5u, generator.sourceSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(21.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(24.0f + d, generator.sourceSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 2; d++) {
		EXPECT_EQ(4.0f + d, generator.targetBatch().at(d, 0));
		EXPECT_EQ(0.0f + d, generator.targetBatch().at(d, 1));
	}

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedClassificationSequenceToSequence) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("features.aligned-feature-reader.label-cache", "labels-1.sequences");
	Core::Configuration::setParameter("source-type", "sequence");
	Core::Configuration::setParameter("target-type", "sequence");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();

	generator.generateBatch(2);
	EXPECT_EQ(4u, generator.sourceSequenceBatch().nTimeframes());
	EXPECT_EQ(4u, generator.targetSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.targetSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(9.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 12; d++) {
		EXPECT_EQ((d == 3 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((d == t + 3 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((d == t - 1 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(t).at(d, 1));
		}
	}

	generator.generateBatch(2);
	EXPECT_EQ(5u, generator.sourceSequenceBatch().nTimeframes());
	EXPECT_EQ(5u, generator.targetSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.targetSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(21.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(24.0f + d, generator.sourceSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 12; d++) {
		EXPECT_EQ((d == 7 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ((d == 8 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((d == t + 7 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((d == t - 2 ? 1.0f : 0.0f), generator.targetSequenceBatch().at(t).at(d, 1));
		}
	}

	Core::Configuration::reset();
}

TEST_F(Test, TestMinibatchGenerator, supervisedRegressionSequenceToSequence) {

	Core::Configuration::setParameter("features.aligned-feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("features.aligned-feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("source-type", "sequence");
	Core::Configuration::setParameter("target-type", "sequence");

	Nn::MinibatchGenerator generator(Nn::supervised);
	generator.initialize();

	generator.generateBatch(2);
	EXPECT_EQ(4u, generator.sourceSequenceBatch().nTimeframes());
	EXPECT_EQ(4u, generator.targetSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
		EXPECT_EQ((t == 0 ? 1u : 2u), generator.targetSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(9.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 2; d++) {
		EXPECT_EQ(6.0f + d, generator.targetSequenceBatch().at(0).at(d, 0));
		for (u32 t = 1; t < 4; t++) {
			EXPECT_EQ((t+3) * 2.0f + d, generator.targetSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-1) * 2.0f + d, generator.targetSequenceBatch().at(t).at(d, 1));
		}
	}

	generator.generateBatch(2);
	EXPECT_EQ(5u, generator.sourceSequenceBatch().nTimeframes());
	EXPECT_EQ(5u, generator.targetSequenceBatch().nTimeframes());
	for (u32 t = 0; t < 4; t++) {
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.sourceSequenceBatch().at(t).nColumns());
		EXPECT_EQ((t < 2 ? 1u : 2u), generator.targetSequenceBatch().at(t).nColumns());
	}
	for (u32 d = 0; d < 3; d++) {
		EXPECT_EQ(21.0f + d, generator.sourceSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(24.0f + d, generator.sourceSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 3.0f + d, generator.sourceSequenceBatch().at(t).at(d, 1));
		}
	}
	for (u32 d = 0; d < 2; d++) {
		EXPECT_EQ(14.0f + d, generator.targetSequenceBatch().at(0).at(d, 0));
		EXPECT_EQ(16.0f + d, generator.targetSequenceBatch().at(1).at(d, 0));
		for (u32 t = 2; t < 5; t++) {
			EXPECT_EQ((t+7) * 2.0f + d, generator.targetSequenceBatch().at(t).at(d, 0));
			EXPECT_EQ((t-2) * 2.0f + d, generator.targetSequenceBatch().at(t).at(d, 1));
		}
	}

	Core::Configuration::reset();
}
