#include "Trainer.hh"
#include "GradientBasedTrainer.hh"
#include <sstream>
#include <iomanip>

using namespace Nn;

/*
 * Trainer
 */
const Core::ParameterEnum Trainer::paramTrainer_("trainer",
		"dummy, forwarder, feed-forward-trainer, rnn-trainer",
		"dummy", "");

Trainer::Trainer() :
		estimator_(0),
		epoch_(0),
		epochLength_(0),
		isInitialized_(false)
{}

Trainer::~Trainer() {
	if (estimator_)
		delete estimator_;
}

void Trainer::initialize(u32 epochLength) {
	if (!isInitialized_) {
		// initialize the network with a maximal memory = 1 for activations/error signals over time
		// (only activations/error signals of most recent time frame are stored)
		network_.initialize();
		estimator_ = Estimator::createEstimator();
		estimator_->initialize(network_);
		epochLength_ = epochLength;
		isInitialized_ = true;
	}
}

void Trainer::finalize() {
	require(estimator_);
	estimator_->finalize();
}

NeuralNetwork& Trainer::network() {
	require(isInitialized_);
	return network_;
}

Estimator& Trainer::estimator() {
	require(estimator_);
	return *estimator_;
}

void Trainer::setEpoch(u32 epoch) {
	epoch_ = epoch;
	if (estimator_)
		estimator_->setEpoch(epoch);
}

/* factory */
Trainer* Trainer::createFramewiseTrainer() {
	Trainer* trainer = 0;
	switch ( Core::Configuration::config(paramTrainer_) ) {
	case forwarder:
		Core::Log::os("Create forwarder.");
		trainer = new Forwarder();
		break;
	case feedForwardTrainer:
		Core::Log::os("Create feed-forward-trainer.");
		trainer = new FeedForwardTrainer();
		break;
	case rnnTrainer:
		std::cerr << "The selected trainer is no frame-wise trainer. Abort." << std::endl;
		exit(1);
		break;
	case dummy:
	default:
		trainer = new DummyTrainer();
		break;
	}
	return trainer;
}

Trainer* Trainer::createSequenceTrainer() {
	Trainer* trainer = 0;
	switch ( Core::Configuration::config(paramTrainer_) ) {
	case forwarder:
		Core::Log::os("Create forwarder.");
		trainer = new Forwarder();
		break;
	case feedForwardTrainer:
		std::cerr << "The selected trainer is no sequence trainer. Abort." << std::endl;
		exit(1);
		break;
	case rnnTrainer:
		Core::Log::os("Create rnn-trainer.");
		trainer = new RnnTrainer();
		break;
	case dummy:
	default:
		trainer = new DummyTrainer();
		break;
	}
	return trainer;
}

/*
 * Forwarder
 */
const Core::ParameterBool Forwarder::paramCacheNeuralNetworkOutput_("cache-neural-network-output", false, "trainer");

const Core::ParameterBool Forwarder::paramLogConfusionMatrix_("log-confusion-matrix", false, "trainer");

Forwarder::Forwarder() :
		Precursor(),
		statistics_(Statistics::baseStatistics),
		cacheNeuralNetworkOutput_(Core::Configuration::config(paramCacheNeuralNetworkOutput_)),
		logConfusionMatrix_(Core::Configuration::config(paramLogConfusionMatrix_)),
		featureWriter_("trainer.feature-writer")
{}

Forwarder::~Forwarder() {
	if (cacheNeuralNetworkOutput_ && (featureWriter_.nWrittenFeatures() > 0))
		featureWriter_.finalize();
}

void Forwarder::initialize(u32 epochLength) {
	Precursor::initialize(epochLength);
	statistics_.initialize(network_);
	if (logConfusionMatrix_) {
		confusionMatrix_.resize(network_.outputLayer().nUnits(), network_.outputLayer().nUnits());
		confusionMatrix_.setToZero();
	}
}

void Forwarder::finalize() {
	statistics_.normalize();
	Core::Log::os("objective function: ") << statistics_.objectiveFunction();
	Core::Log::os("classification error rate: ") << (Float)statistics_.nClassificationErrors() / statistics_.nObservations()
			<< " (" << statistics_.nClassificationErrors() << "/" << statistics_.nObservations() << ")";
	// log confusion matrix
	if (logConfusionMatrix_) {
		Core::Log::openTag("confusion-matrix", "(true classes x predicted classes)");
		std::stringstream s;
		s << std::setfill(' ') << std::setw(7) << "class";
		for (u32 i = 0; i < confusionMatrix_.nRows(); i++) {
			s << std::setfill(' ') << std::setw(7) << i;
		}
		Core::Log::os() << s.str();
		for (u32 i = 0; i < confusionMatrix_.nRows(); i++) {
			s.str("");
			s << std::setfill(' ') << std::setw(7) << i;
			for (u32 j = 0; j < confusionMatrix_.nColumns(); j++) {
				s << std::setfill(' ') << std::setw(7) << confusionMatrix_.at(i,j);
			}
			Core::Log::os() << s.str();
		}
		Core::Log::closeTag();
	}
	Precursor::finalize();
}

void Forwarder::updateStatistics(LabelVector& labels) {
	// ensure that network and labels are in computing state
	labels.initComputation();
	network_.initComputation(false); // no synchronization necessary
	// update the statistics
	Float objectiveFunction = network_.outputLayer().latestActivations(0).crossEntropyObjectiveFunction(labels);
	u32 nClassificationErrors = network_.outputLayer().latestActivations(0).nClassificationErrors(labels);
	u32 nObservations = network_.outputLayer().latestActivations(0).nColumns();
	statistics_.increaseNumberOfObservations(nObservations);
	statistics_.increaseNumberOfClassificationErrors(nClassificationErrors);
	statistics_.addToObjectiveFunction(objectiveFunction);
	Core::Log::os("batch objective function: ") << objectiveFunction / nObservations;
	Core::Log::os("batch classification error rate: ") << (Float)nClassificationErrors /  nObservations;
	if (logConfusionMatrix_) {
		labels.finishComputation(false);
		Math::CudaVector<u32> argMax(network_.outputLayer().latestActivations(0).nColumns());
		argMax.initComputation();
		network_.outputLayer().latestActivations(0).argMax(argMax);
		argMax.finishComputation();
		for (u32 i = 0; i < argMax.nRows(); i++) {
			confusionMatrix_.at(labels.at(i), argMax(i)) += 1;
		}
	}
}

void Forwarder::cacheNetworkOutput() {
	// convert output
	network_.outputLayer().latestActivations(0).finishComputation();
	Math::Matrix<Float> output(network_.outputLayer().latestActivations(0).nRows(),
			network_.outputLayer().latestActivations(0).nColumns());
	for (u32 i = 0; i < output.nRows(); i++) {
		for (u32 j = 0; j < output.nColumns(); j++) {
			output.at(i, j) = network_.outputLayer().latestActivations(0).at(i, j);
		}
	}
	featureWriter_.write(output);
}

void Forwarder::processBatch(Matrix& batch) {
	network_.forward(batch);
	if (cacheNeuralNetworkOutput_) {
		cacheNetworkOutput();
	}
}

void Forwarder::processBatch(Matrix& batch, LabelVector& labels) {
	labels.initComputation();
	processBatch(batch);
	// make sure latest activations are in computing state again
	// (may not be the case if network output has been cached)
	network_.outputLayer().latestActivations(0).initComputation(false);
	updateStatistics(labels);
}

void Forwarder::processSequenceBatch(MatrixContainer& batchedSequence) {
	// ensure that the previous activations are stored
	// (for standard RNNs 2 is sufficient but for special layers (e.g. attention) all activations are required)
	network_.setMaximalMemory(batchedSequence.nTimeframes());
	network_.forwardSequence(batchedSequence);
	if (cacheNeuralNetworkOutput_) {
		cacheNetworkOutput();
	}
}

void Forwarder::processSequenceBatch(MatrixContainer& batchedSequence, LabelVector& labels) {
	labels.initComputation();
	processSequenceBatch(batchedSequence);
	updateStatistics(labels);
}
