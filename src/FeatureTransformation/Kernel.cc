#include "Kernel.hh"
#include "MultiChannelRbfChiSquareKernel.hh"

using namespace FeatureTransformation;

const Core::ParameterEnum Kernel::paramKernel_("kernel", "none, linear, histogram-intersection, hellinger,"
		"multichannel-rbf-chi-square, modified-chi-square",
		"none");

Kernel::Kernel() :
		featureReaderTrain_("feature-transformation.kernel.train"),
		featureReaderTest_("feature-transformation.kernel.test"),
		isInitialized_(false),
		isFinalized_(false)
{}

void Kernel::initialize() {
	if (!isInitialized_) {
		featureReaderTrain_.initialize();
		featureReaderTest_.initialize();
		isInitialized_ = true;
	}
}

void Kernel::finalize() {
	if (!isFinalized_) {
		featureWriter_.finalize();
		isFinalized_ = true;
	}
}

void Kernel::applyKernel() {
	require(isInitialized_);
	while (featureReaderTest_.hasFeatures()) {
		Math::Vector<Float> output;
		const Math::Vector<Float>& input = featureReaderTest_.next();
		applyKernel(input, output);
		featureWriter_.write(output);
	}
}

Kernel* Kernel::createKernel() {

	Kernel* kernel = 0;

	switch (Core::Configuration::config(paramKernel_)) {
	case linear:
		{
		kernel = new LinearKernel;
		Core::Log::os("Create linear kernel.");
		}
		break;
	case histogramIntersection:
		{
		kernel = new HistogramIntersectionKernel;
		Core::Log::os("Create histogram-intersection kernel.");
		}
		break;
	case hellinger:
		{
		kernel = new HellingerKernel;
		Core::Log::os("Create hellinger kernel.");
		}
		break;
	case multichannelRbfChiSquare:
		{
		kernel = new MultiChannelRbfChiSquareKernel;
		Core::Log::os("Create multichannel-rbf-chi-square kernel.");
		}
		break;
	case modifiedChiSquare:
		{
		kernel = new ModifiedChiSquareKernel;
		Core::Log::os("Create modified-chi-square kernel.");
		}
		break;
	case none:
	default:
		std::cerr << "No kernel chosen. Abort." << std::endl;
		exit(1);
	}

	return kernel;
}

/*
 * LinearKernel
 */
void LinearKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) {
	require(isInitialized_);
	output.resize(featureReaderTrain_.totalNumberOfFeatures());

	featureReaderTrain_.newEpoch();
	u32 i = 0;
	while (featureReaderTrain_.hasFeatures()) {
		output.at(i) = input.dot(featureReaderTrain_.next());
		i++;
	}
}

/*
 * HistogramIntersectionKernel
 */
void HistogramIntersectionKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) {
	require(isInitialized_);
	output.resize(featureReaderTrain_.totalNumberOfFeatures());

	featureReaderTrain_.newEpoch();
	u32 i = 0;
	while (featureReaderTrain_.hasFeatures()) {
		const Math::Vector<Float>& f = featureReaderTrain_.next();
		Math::Vector<Float> tmp(f.size());
#pragma omp parallel for
		for (u32 d = 0; d < f.size(); d++) {
			tmp.at(d) = std::min(input.at(d), f.at(d));
		}
		output.at(i) = tmp.l1norm();
		i++;
	}
}

/*
 * HellingerKernel
 */
void HellingerKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) {
	require(isInitialized_);
	output.resize(featureReaderTrain_.totalNumberOfFeatures());

	featureReaderTrain_.newEpoch();
	u32 i = 0;
	while (featureReaderTrain_.hasFeatures()) {
		const Math::Vector<Float>& f = featureReaderTrain_.next();
		Math::Vector<Float> tmp(f.size());
#pragma omp parallel for
		for (u32 d = 0; d < f.size(); d++) {
			tmp.at(d) = std::sqrt(input.at(d) * f.at(d));
		}
		output.at(i) = tmp.l1norm();
		i++;
	}
}

/*
 * ModifiedChiSquareKernel
 */
void ModifiedChiSquareKernel::applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) {
	require(isInitialized_);
	output.resize(featureReaderTrain_.totalNumberOfFeatures());

	featureReaderTrain_.newEpoch();
	u32 i = 0;
	while (featureReaderTrain_.hasFeatures()) {
		output.at(i) = input.modifiedChiSquareDistance(featureReaderTrain_.next());
		i++;
	}
}
