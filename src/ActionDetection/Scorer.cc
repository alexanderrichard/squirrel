#include "Scorer.hh"

using namespace ActionDetection;

/*
 * Scorer
 */
const Core::ParameterEnum Scorer::paramScorerType_("type", "none, neural-network-scorer", "none", "scorer");

Scorer::Scorer() :
		sequence_(0),
		hasSequence_(false),
		isInitialized_(false)
{}

void Scorer::initialize() {
	isInitialized_ = true;
}

void Scorer::setSequence(const Math::Matrix<Float>& sequence) {
	sequence_ = &sequence;
	hasSequence_ = true;
}

Scorer* Scorer::create() {
	Scorer* p = 0;
	switch ((ScorerType)Core::Configuration::config(paramScorerType_)) {
	case neuralNetworkScorer:
		p = new NeuralNetworkScorer();
		Core::Log::os("Created neural-network-scorer.");
		break;
	case none:
	default:
		std::cerr << "Error: Scorer.cc: No scorer type specified. Abort." << std::endl;
		exit(1);
	}
	return p;
}

/*
 * NeuralNetworkScorer
 */
const Core::ParameterString NeuralNetworkScorer::paramPriorFile_("prior-file", "", "scorer");

const Core::ParameterFloat NeuralNetworkScorer::paramPriorScale_("prior-scale", 1.0, "scorer");

const Core::ParameterInt NeuralNetworkScorer::paramBatchSize_("batch-size", 1, "scorer");

NeuralNetworkScorer::NeuralNetworkScorer() :
		Precursor(),
		priorFile_(Core::Configuration::config(paramPriorFile_)),
		priorScale_(Core::Configuration::config(paramPriorScale_)),
		batchSize_(Core::Configuration::config(paramBatchSize_)),
		t_start_(Types::max<u32>()),
		t_end_(Types::max<u32>())
{
	require_gt(batchSize_, 0);
}

void NeuralNetworkScorer::generateVector(u32 t_start, u32 t_end, u32 column, Math::CudaMatrix<Float>& result) {
	require_le(t_start, t_end);
	require_eq(result.nRows(), sequence_->nRows());
	require_lt(column, result.nColumns());
#pragma omp parallel for
	for (u32 row = 0; row < sequence_->nRows(); row++) {
		result.at(row, column) = sequence_->at(row, t_end);
		if (t_start > 0) {
			result.at(row, column) -= sequence_->at(row, t_start);
		}
	}
}

void NeuralNetworkScorer::initialize() {
	Core::Log::openTag("scorer");
	// initialize neural network
	network_.initialize();
	// read prior
	if (priorFile_.empty()) {
		Core::Log::os("No prior file given. Assume uniform prior.");
		prior_.resize(network_.outputDimension());
		prior_.fill(1.0 / network_.outputDimension());
	}
	else {
		Core::Log::os("Read prior file from ") << priorFile_ << ".";
		prior_.read(priorFile_);
		require_eq(prior_.size(), network_.outputDimension());
		for (u32 i = 0; i < prior_.size(); i++) {
			require_le(prior_.at(i), 0);
		}
		Core::Log::os("Use prior scale ") << priorScale_ << ".";
	}
	prior_.scale(priorScale_);
	networkInput_.resize(network_.inputDimension(), batchSize_);
	Core::Log::closeTag();
	isInitialized_ = true;
}

void NeuralNetworkScorer::setSequence(const Math::Matrix<Float>& sequence) {
	Precursor::setSequence(sequence);
	require_eq(sequence_->nRows(), network_.inputDimension());
}

Float NeuralNetworkScorer::score(u32 c, u32 t_start, u32 t_end) {
	require(isInitialized_);
	u32 T = sequence_->nColumns();
	require_lt(t_end, T);
	require_le(t_start, t_end);
	// are the scores already precomputed or do we have to compute them first?
	if ((t_start < t_start_) || (t_start > t_start_ + batchSize_ - 1) || (t_end != t_end_)) {
		for (u32 column = 0; column < std::min(batchSize_, (t_end - t_start + 1)); column++) {
			generateVector(t_start + column, t_end, column, networkInput_);
		}
		network_.forward(networkInput_);
		networkInput_.finishComputation(false);
		network_.outputLayer().finishComputation(true);

		// update for which t and t_start the scores are precomputed
		t_start_ = t_start;
		t_end_ = t_end;
	}

	u32 column = t_start - t_start_;
	return std::log(network_.outputLayer().latestActivations(0).at(c, column)) - prior_.at(c);
}

u32 NeuralNetworkScorer::nClasses() const {
	require(isInitialized_);
	return network_.outputDimension();
}
