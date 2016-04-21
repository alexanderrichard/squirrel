#include <Test/UnitTest.hh>
#include <Nn/NeuralNetwork.hh>
#include <Nn/MatrixContainer.hh>
#include "Nn_TestHelpers.hh"

class TestNeuralNetwork : public Test::Fixture
{
public:
	void setUp();
	void tearDown();
};

void TestNeuralNetwork::setUp() {
}

void TestNeuralNetwork::tearDown() {
}

TEST_F(Test, TestNeuralNetwork, weightsIO)
{
	Core::Configuration::setParameter("neural-network.conn-0-1.write-weights-to", "__tmp_conn-0-1__");
	Core::Configuration::setParameter("neural-network.conn-1-1.write-weights-to", "__tmp_conn-1-1__");
	Core::Configuration::setParameter("neural-network.conn-1-2.write-weights-to", "__tmp_conn-1-2__");
	Core::Configuration::setParameter("neural-network.conn-2-2.write-weights-to", "__tmp_conn-2-2__");
	Core::Configuration::setParameter("neural-network.layer-1.write-bias-to", "__tmp_layer-1__");
	Core::Configuration::setParameter("neural-network.layer-2.write-bias-to", "__tmp_layer-2__");

	Nn::NeuralNetwork* reference = configureRecurrentNetwork();
	reference->initialize(2);
	modifyRecurrentNetwork(reference);
	reference->saveNeuralNetworkParameters();

	Core::Configuration::setParameter("neural-network.conn-0-1.load-weights-from", "__tmp_conn-0-1__");
	Core::Configuration::setParameter("neural-network.conn-1-1.load-weights-from", "__tmp_conn-1-1__");
	Core::Configuration::setParameter("neural-network.conn-1-2.load-weights-from", "__tmp_conn-1-2__");
	Core::Configuration::setParameter("neural-network.conn-2-2.load-weights-from", "__tmp_conn-2-2__");
	Core::Configuration::setParameter("neural-network.layer-1.load-bias-from", "__tmp_layer-1__");
	Core::Configuration::setParameter("neural-network.layer-2.load-bias-from", "__tmp_layer-2__");

	Nn::NeuralNetwork* network = configureRecurrentNetwork();
	network->initialize(2);
	modifyRecurrentNetwork(network);

	for (u32 l = 0; l < 2; l++) {
		for (u32 c = 0; c < 2; c++) {
			for (u32 i = 0; i < network->layer(l).weights(c,0).nRows(); i++) {
				for (u32 j = 0; j < network->layer(l).weights(c,0).nColumns(); j++) {
					EXPECT_EQ(network->layer(l).weights(c,0).at(i,j), reference->layer(l).weights(c,0).at(i,j));
				}
			}
		}
		for (u32 i = 0; i < network->layer(l).bias(0).nRows(); i++) {
			EXPECT_EQ(network->layer(l).bias(0).at(i), reference->layer(l).bias(0).at(i));
		}
	}

	delete reference;
	delete network;

	remove("__tmp_conn-0-1__");
	remove("__tmp_conn-1-1__");
	remove("__tmp_conn-1-2__");
	remove("__tmp_conn-2-2__");
	remove("__tmp_layer-1__");
	remove("__tmp_layer-2__");

	Core::Configuration::reset();
}

TEST_F(Test, TestNeuralNetwork, forward)
{
	Nn::Matrix input(2,2);
	input.at(0,0) = -2;
	input.at(1,0) = 0.5;
	input.at(0,1) = 2;
	input.at(1,1) = 0;

	Nn::NeuralNetwork* network = configureFeedForwardNetwork();
	network->initialize();
	modifyFeedForwardNetwork(network);

	network->forward(input);
	network->finishComputation();

	EXPECT_EQ(network->outputLayer().latestActivations(0).nRows(), 3u);
	EXPECT_EQ(network->outputLayer().latestActivations(0).nColumns(), 2u);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(0,0), (f32)0.28887157, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(1,0), (f32)0.39445432, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(2,0), (f32)0.31667411, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(0,1), (f32)0.15791667, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(1,1), (f32)0.68274409, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(2,1), (f32)0.15933924, 0.000001);

	delete network;
	Core::Configuration::reset();
}

TEST_F(Test, TestNeuralNetwork, forwardSequence) {
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

	Nn::NeuralNetwork* network = configureRecurrentNetwork();
	network->initialize(2);
	modifyRecurrentNetwork(network);

	network->forwardSequence(inputSequence);
	network->finishComputation();
	EXPECT_EQ(network->outputLayer().latestActivations(0).nRows(), 3u);
	EXPECT_EQ(network->outputLayer().latestActivations(0).nColumns(), 2u);
	// output for sequence 1
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(0,0), (f32)0.31190455, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(1,0), (f32)0.54252862, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(2,0), (f32)0.14556683, 0.000001);
	// output for sequence 2
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(0,1), (f32)0.57369623, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(1,1), (f32)0.26263739, 0.000001);
	EXPECT_DOUBLE_EQ(network->outputLayer().latestActivations(0).at(2,1), (f32)0.16366638, 0.000001);

	delete network;
	Core::Configuration::reset();
}
