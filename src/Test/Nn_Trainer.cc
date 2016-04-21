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
#include <Nn/GradientBasedTrainer.hh>
#include <Nn/Types.hh>
#include "Nn_TestHelpers.hh"

using namespace std;

class TestNnTrainer : public Test::Fixture
{
public:
	void setUp();
	void tearDown();
};

void TestNnTrainer::setUp() {
}

void TestNnTrainer::tearDown() {
}

TEST_F(Test, TestNnTrainer, CrossEntropyFeedForwardTrainer_processBatch) {

	Core::Configuration::setParameter("trainer", "feed-forward-trainer");
	Core::Configuration::setParameter("training-criterion", "cross-entropy");
	Core::Configuration::setParameter("trainer.model-update-strategy", "after-epoch");
	Core::Configuration::setParameter("estimator.method", "steepest-descent");

	Nn::Matrix input(2,2);
	input.at(0,0) = -2;
	input.at(1,0) = 0.5;
	input.at(0,1) = 2;
	input.at(1,1) = 0;
	// create label vector
	Nn::LabelVector labels;
	labels.resize(2);
	labels.setToZero();
	labels.at(0) = 0;
	labels.at(1) = 2;

	Nn::NeuralNetwork* dummy = configureFeedForwardNetwork(); // set config parameters
	delete dummy;
	Nn::FeedForwardTrainer* trainer = new Nn::FeedForwardTrainer();

	trainer->initialize(100); // set epoch length to 100
	modifyFeedForwardNetwork(&(trainer->network()));
	trainer->processBatch(input, labels);
	trainer->statistics().normalize();
	trainer->network().finishComputation();

	// error signal l=1
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(0,0), (f32)-0.71112843, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(1,0), (f32)0.39445432, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(2,0), (f32)0.31667411, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(0,1), (f32)0.15791667, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(1,1), (f32)0.68274409, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(2,1), (f32)-0.84066076, 0.000001);
	// error signal l=0
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(0,0), (f32)0.1239779, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(1,0), (f32)-0.01740096, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(0,1), (f32)3.15423886e-02, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(1,1), (f32)-3.81639031e-05, 0.000001);

	/* gradient */
	trainer->statistics().finishComputation();
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(0,0), (f32)-0.09243551, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(0,1), (f32)0.0173628, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(1,0), (f32)0.03099447, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(1,1), (f32)-0.00435024, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-1", 0).at(0), (f32)0.07776014, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-1", 0).at(1), (f32)-0.00871956, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,0), (f32)-0.01808777, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,1), (f32)0.38827461, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,2), (f32)-0.37018684, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,0), (f32)-0.21174321, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,1), (f32)0.50261423, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,2), (f32)-0.29087102, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(0), (f32)-0.27660588, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(1), (f32)0.5385992, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(2), (f32)-0.26199332, 0.000001);

	delete trainer;
	Core::Configuration::reset();
}

TEST_F(Test, TestNnTrainer, CrossEntropyRnnTrainer_processSequenceBatch) {

	Core::Configuration::setParameter("trainer", "rnn-trainer");
	Core::Configuration::setParameter("training-criterion", "cross-entropy");
	Core::Configuration::setParameter("trainer.model-update-strategy", "after-epoch");
	Core::Configuration::setParameter("estimator.method", "steepest-descent");

	// input: two sequences (mini-batch of sequences) with two time frames each
	Nn::MatrixContainer inputSequence;
	inputSequence.setMaximalMemory(2);
	inputSequence.addTimeframe(2,2);
	inputSequence.addTimeframe(2,2);
	// first sequence
	inputSequence.at(0).at(0,0) = -2;
	inputSequence.at(0).at(1,0) = 0.5;
	inputSequence.at(1).at(0,0) = 2;
	inputSequence.at(1).at(1,0) = 0;
	// second sequence
	inputSequence.at(0).at(0,1) = -1.0;
	inputSequence.at(0).at(1,1) = -0.5;
	inputSequence.at(1).at(0,1) = 0.5;
	inputSequence.at(1).at(1,1) = 1.0;
	// create label vector
	Nn::LabelVector labels;
	labels.resize(2);
	labels.setToZero();
	labels.at(0) = 2;
	labels.at(1) = 1;

	Nn::NeuralNetwork* dummy = configureRecurrentNetwork(); // set config parameters
	delete dummy;
	Nn::RnnTrainer* trainer = new Nn::RnnTrainer();

	trainer->initialize(100); // set epoch-length to 100
	modifyRecurrentNetwork(&(trainer->network()));
	trainer->processSequenceBatch(inputSequence, labels);
	trainer->statistics().normalize();
	trainer->network().finishComputation();

	/* output for sequence 1 */
	// error signal l=1, t=1
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(0,0), (f32)0.31190455, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(1,0), (f32)0.54252862, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(2,0), (f32)-0.85443317, 0.000001);
	// error signal l=0, t=1
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(1,0).at(0,0), (f32)0.03039778, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(1,0).at(1,0), (f32)-0.00018048, 0.000001);
	// error signal l=1, t=0
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(0,0), (f32)0.20988482, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(1,0), (f32)0.3481139, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(2,0), (f32)-0.55799872, 0.000001);
	// error signal l=0, t=0
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(0,0), (f32)0.18869979, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(1,0), (f32)-0.20264793, 0.000001);

	/* output for sequence 2 */
	// error signal l=1, t=1
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(0,1), (f32)0.57369623, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(1,1), (f32)-0.73736261, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(1,0).at(2,1), (f32)0.16366638, 0.000001);
	// error signal l=0, t=1
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(1,0).at(0,1), (f32)-0.38879422, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(1,0).at(1,1), (f32)0.02824409, 0.000001);
	// error signal l=1, t=0
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(0,1), (f32)0.00501267, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(1,1), (f32)0.19908168, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(1).errorSignal(0,0).at(2,1), (f32)-0.20409436, 0.000001);
	// error signal l=0, t=0
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(0,1), (f32)0.07737752, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->network().layer(0).errorSignal(0,0).at(1,1), (f32)-0.00101234, 0.000001);

	/* gradient */
	trainer->statistics().finishComputation();
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(0,0), (f32)-0.58837864 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(0,1), (f32)0.42006928 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(1,0), (f32)-0.33313309 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-0-1").at(1,1), (f32)-0.07257371 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-1").at(0,0), (f32)-0.05151153 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-1").at(0,1), (f32)-0.0410261 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-1").at(1,0), (f32)-0.1185594 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-1").at(1,1), (f32)-0.13923009 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-1", 0).at(0), (f32)-0.09231913 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-1", 0).at(1), (f32)-0.17559666 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,0), (f32)0.66175943 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,1), (f32)0.42070271 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(0,2), (f32)-1.08246214 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,0), (f32)1.04992504 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,1), (f32)0.30425651 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-1-2").at(1,2), (f32)-1.35418155 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(0,0), (f32)0.48793701 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(0,1), (f32)-0.117682 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(0,2), (f32)-0.37025501 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(1,0), (f32)0.40585015 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(1,1), (f32)0.3638503 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(1,2), (f32)-0.76970045 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(2,0), (f32)0.20671112 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(2,1), (f32)0.10619329 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().weightsGradient("conn-2-2").at(2,2), (f32)-0.31290441 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(0), (f32)1.10049828 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(1), (f32)0.35236159 / 2.0, 0.000001);
	EXPECT_DOUBLE_EQ(trainer->statistics().biasGradient("layer-2", 0).at(2), (f32)-1.45285987 / 2.0, 0.000001);

	delete trainer;
	Core::Configuration::reset();
}
