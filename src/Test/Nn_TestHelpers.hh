#ifndef TEST_NN_TESTHELPERS_HH_
#define TEST_NN_TESTHELPERS_HH_

#include <Nn/NeuralNetwork.hh>

Nn::NeuralNetwork* configureFeedForwardNetwork();
void modifyFeedForwardNetwork(Nn::NeuralNetwork* network);

Nn::NeuralNetwork* configureRecurrentNetwork();
void modifyRecurrentNetwork(Nn::NeuralNetwork* network);

#endif /* TEST_NN_TESTHELPERS_HH_ */
