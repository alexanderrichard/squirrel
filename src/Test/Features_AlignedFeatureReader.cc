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
 * Features_AlignedFeatureReader.cc
 *
 *  Created on: Jun 2, 2016
 *      Author: richard
 */

#include <Test/UnitTest.hh>
#include <sstream>
#include <stdlib.h>
#include <Math/Random.hh>
#include <Features/AlignedFeatureReader.hh>

class TestAlignedFeatureReader : public Test::Fixture
{
public:
	void setUp();
	void tearDown() {}
	void testAlignedFeatureReader(Features::AlignedFeatureReader* featureReader);
	void testAlignedSequenceFeatureReader(Features::AlignedSequenceFeatureReader* featureReader);
	void testTemporallyAlignedSequenceFeatureReader(Features::TemporallyAlignedSequenceFeatureReader* featureReader);
};

void TestAlignedFeatureReader::setUp() {
	Core::Configuration::setParameter("math.random.seed", "10");
	Math::Random::initializeSRand();
}

void TestAlignedFeatureReader::testAlignedFeatureReader(Features::AlignedFeatureReader* featureReader) {

	featureReader->initialize();
	u32 bufferSize = featureReader->maxBufferSize();
	EXPECT_EQ(12u, featureReader->totalNumberOfFeatures());
	EXPECT_EQ(3u, featureReader->featureDimension());
	if (dynamic_cast<const Features::LabeledFeatureReader*>(featureReader) == 0) {
		EXPECT_EQ(2u, featureReader->targetDimension());
	}
	else {
		EXPECT_EQ(12u, featureReader->targetDimension());
	}
	// test two epochs
	for (u32 epoch = 0; epoch < 2; epoch++) {
		// test the current epoch
		for (u32 i = 0; i < featureReader->totalNumberOfFeatures(); i++) {
			const Math::Vector<Float>& f = featureReader->next();
			// get index of observation in cache
			u32 j = i;
			if (featureReader->shuffleBuffer())
				j = i - (i % bufferSize) + featureReader->getReordering().at(i % bufferSize);
			for (u32 d = 0; d < featureReader->featureDimension(); d++)
				EXPECT_EQ(j*3.0f + d, f.at(d));
			if (dynamic_cast<const Features::LabeledFeatureReader*>(featureReader) == 0) {
				const Math::Vector<Float>& g = featureReader->target();
				for (u32 d = 0; d < featureReader->targetDimension(); d++)
					EXPECT_EQ(j*2.0f + d, g.at(d));
			}
			else { // labelCache
				EXPECT_EQ(j, dynamic_cast< Features::LabeledFeatureReader* >(featureReader)->label());
			}
		}
		EXPECT_FALSE(featureReader->hasFeatures());
		featureReader->newEpoch();
	}
}

void TestAlignedFeatureReader::testAlignedSequenceFeatureReader(Features::AlignedSequenceFeatureReader* featureReader) {

	featureReader->initialize();
	u32 bufferSize = featureReader->maxBufferSize();
	EXPECT_EQ(12u, featureReader->totalNumberOfFeatures());
	EXPECT_EQ(3u, featureReader->featureDimension());
	EXPECT_EQ(3u, featureReader->totalNumberOfSequences());
	if (dynamic_cast<const Features::LabeledSequenceFeatureReader*>(featureReader) == 0) {
		EXPECT_EQ(2u, featureReader->targetDimension());
	}
	else {
		EXPECT_EQ(3u, featureReader->targetDimension());
	}
	// test two epochs
	for (u32 epoch = 0; epoch < 2; epoch++) {
		// test the current epoch
		for (u32 i = 0; i < featureReader->totalNumberOfSequences(); i++) {
			u32 nFeatures = 0;
			const Math::Matrix<Float>& f = featureReader->next();
			// get index of observation in cache
			u32 j = i;
			if (featureReader->shuffleBuffer() || featureReader->areSequencesSorted())
				j = i - (i % bufferSize) + featureReader->getReordering().at(i % bufferSize);
			// check sequence length
			EXPECT_EQ(j + 3, f.nColumns());
			// check features
			u32 offset = 0; // j = 0
			if (j == 1) offset = 9;
			if (j == 2) offset = 21;
			for (u32 col = 0; col < f.nColumns(); col++) {
				EXPECT_EQ(offset + col * 3.0f + 0.0f, f.at(0, col));
				EXPECT_EQ(offset + col * 3.0f + 1.0f, f.at(1, col));
				EXPECT_EQ(offset + col * 3.0f + 2.0f, f.at(2, col));
			}
			if (dynamic_cast<const Features::LabeledSequenceFeatureReader*>(featureReader) == 0) {
				const Math::Vector<Float>& g = featureReader->target();
				for (u32 d = 0; d < featureReader->targetDimension(); d++)
					EXPECT_EQ(j*2.0f + d, g.at(d));
			}
			else { // labelCache
				EXPECT_EQ(j, dynamic_cast< Features::LabeledSequenceFeatureReader* >(featureReader)->label());
			}
			nFeatures += f.nColumns() * f.nRows();
		}
		EXPECT_FALSE(featureReader->hasSequences());
		featureReader->newEpoch();
	}
}

void TestAlignedFeatureReader::testTemporallyAlignedSequenceFeatureReader(Features::TemporallyAlignedSequenceFeatureReader* featureReader) {

	featureReader->initialize();
	u32 bufferSize = featureReader->maxBufferSize();
	EXPECT_EQ(12u, featureReader->totalNumberOfFeatures());
	EXPECT_EQ(3u, featureReader->featureDimension());
	EXPECT_EQ(3u, featureReader->totalNumberOfSequences());
	if (dynamic_cast<const Features::TemporallyLabeledSequenceFeatureReader*>(featureReader) == 0) {
		EXPECT_EQ(2u, featureReader->targetDimension());
	}
	else {
		EXPECT_EQ(12u, featureReader->targetDimension());
	}
	// test two epochs
	for (u32 epoch = 0; epoch < 2; epoch++) {
		// test the current epoch
		for (u32 i = 0; i < featureReader->totalNumberOfSequences(); i++) {
			const Math::Matrix<Float>& f = featureReader->next();
			// get index of observation in cache
			u32 j = i;
			if (featureReader->shuffleBuffer() || featureReader->areSequencesSorted())
				j = i - (i % bufferSize) + featureReader->getReordering().at(i % bufferSize);
			// check sequence length
			EXPECT_EQ(j + 3, f.nColumns());
			// check features
			u32 offset = 0; // j = 0
			if (j == 1) offset = 3;
			if (j == 2) offset = 7;
			for (u32 col = 0; col < f.nColumns(); col++) {
				EXPECT_EQ((offset + col) * 3.0f + 0.0f, f.at(0, col));
				EXPECT_EQ((offset + col) * 3.0f + 1.0f, f.at(1, col));
				EXPECT_EQ((offset + col) * 3.0f + 2.0f, f.at(2, col));
			}
			if (dynamic_cast<const Features::TemporallyLabeledSequenceFeatureReader*>(featureReader) == 0) {
				const Math::Matrix<Float>& g = featureReader->target();
				for (u32 col = 0; col < g.nColumns(); col++) {
					EXPECT_EQ((offset + col) * 2.0f + 0.0f, g.at(0, col));
					EXPECT_EQ((offset + col) * 2.0f + 1.0f, g.at(1, col));
				}
			}
			else { // labelCache
				const std::vector<u32>& g = dynamic_cast< Features::TemporallyLabeledSequenceFeatureReader* >(featureReader)->labelSequence();
				for (u32 col = 0; col < g.size(); col++) {
					EXPECT_EQ(offset + col, g.at(col));
				}
			}
		}
		EXPECT_FALSE(featureReader->hasSequences());
		featureReader->newEpoch();
	}
}

/* tests for aligned feature reader */
TEST_F(Test, TestAlignedFeatureReader, alignedFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.vectors");
	Features::AlignedFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);

	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, alignedFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::AlignedFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, alignedFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.vectors");
	Core::Configuration::setParameter("feature-reader.buffer-size", "5");
	Features::AlignedFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, alignedFeatureReaderSmallBufferShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "5");
	Features::AlignedFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

/* tests for labeled feature reader */
TEST_F(Test, TestAlignedFeatureReader, labeledFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.vectors");
	Features::LabeledFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::LabeledFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.vectors");
	Core::Configuration::setParameter("feature-reader.buffer-size", "5");
	Features::LabeledFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledFeatureReaderSmallBufferShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.vectors");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "5");
	Features::LabeledFeatureReader featureReader("feature-reader");
	testAlignedFeatureReader(&featureReader);
	Core::Configuration::reset();
}

/* tests for aligned sequence feature reader */
TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReaderSorted) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("feature-reader.sort-sequences", "true");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReaderShuffledSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, aligendSequenceFeatureReaderSortedSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-2.vectors");
	Core::Configuration::setParameter("feature-reader.sort-sequences", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::AlignedSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

/* tests for labeled sequence feature reader */
TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReaderSorted) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Core::Configuration::setParameter("feature-reader.sort-sequences", "true");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReaderShuffledSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, labeledSequenceFeatureReaderSortedSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-2.vectors");
	Core::Configuration::setParameter("feature-reader.sort-sequences", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::LabeledSequenceFeatureReader featureReader("feature-reader");
	testAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

/* tests for temporally aligned sequence feature reader */
TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReaderSorted) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("feature-reader.sort-sequence", "true");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReaderShuffledSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyAligendSequenceFeatureReaderSortedSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "targets-1.sequences");
	Core::Configuration::setParameter("feature-reader.sort-sequence", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyAlignedSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

/* tests for temporally labeled sequence feature reader */
TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReader) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReaderShuffled) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReaderSorted) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Core::Configuration::setParameter("feature-reader.sort-sequence", "true");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReaderSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReaderShuffledSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Core::Configuration::setParameter("feature-reader.shuffle-buffer", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}

TEST_F(Test, TestAlignedFeatureReader, temporallyLabeledSequenceFeatureReaderSortedSmallBuffer) {
	Core::Configuration::setParameter("feature-reader.feature-cache", "input.sequences");
	Core::Configuration::setParameter("feature-reader.target-cache", "labels-1.sequences");
	Core::Configuration::setParameter("feature-reader.sort-sequence", "true");
	Core::Configuration::setParameter("feature-reader.buffer-size", "2");
	Features::TemporallyLabeledSequenceFeatureReader featureReader("feature-reader");
	testTemporallyAlignedSequenceFeatureReader(&featureReader);
	Core::Configuration::reset();
}
