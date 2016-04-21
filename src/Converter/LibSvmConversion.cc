#include "LibSvmConversion.hh"
#include "Math/Vector.hh"

using namespace Converter;

const Core::ParameterString LibSvmConverter::paramLibSvmFile_("lib-svm-file", "", "converter.lib-svm-conversion");

const Core::ParameterBool LibSvmConverter::paramUsePrecomputedKernelFormat_("use-precomputed-kernel-format", false,
		"converter.lib-svm-conversion");

LibSvmConverter::LibSvmConverter() :
		libSvmFile_(Core::Configuration::config(paramLibSvmFile_)),
		usePrecomputedKernelFormat_(Core::Configuration::config(paramUsePrecomputedKernelFormat_))
{}

void LibSvmConverter::convert() {
	Core::AsciiStream libSvmStream(libSvmFile_, std::ios::out);
	featureReader_.initialize();
	u32 index = 0;
	while (featureReader_.hasFeatures()) {
		const Math::Vector<Float>& v = featureReader_.next();
		libSvmStream << featureReader_.label();
		if (usePrecomputedKernelFormat_) {
			libSvmStream << " " << "0:" << index+1;
		}
		for (u32 i = 0; i < featureReader_.featureDimension(); i++) {
			libSvmStream << " " << i+1 << ":" << v.at(i);
		}
		libSvmStream << Core::IOStream::endl;
		index++;
	}
	libSvmStream.close();
}

/*
 * LibSvmToLogLinear
 */
const Core::ParameterString LibSvmToLogLinear::paramRhoFile_("rho-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramSupportVectorCoefficientsFile_("support-vector-coefficients-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramSupportVectorIndexFile_("support-vector-index-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramProbAFile_("probA-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramProbBFile_("probB-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramBiasFile_("bias-file", "", "converter.lib-svm-to-log-linear");

const Core::ParameterString LibSvmToLogLinear::paramWeightsFile_("weights-file", "", "converter.lib-svm-to-log-linear");

LibSvmToLogLinear::LibSvmToLogLinear() :
		rhoFile_(Core::Configuration::config(paramRhoFile_)),
		supportVectorCoefficientsFile_(Core::Configuration::config(paramSupportVectorCoefficientsFile_)),
		supportVectorIndexFile_(Core::Configuration::config(paramSupportVectorIndexFile_)),
		probAFile_(Core::Configuration::config(paramProbAFile_)),
		probBFile_(Core::Configuration::config(paramProbBFile_)),
		biasFile_(Core::Configuration::config(paramBiasFile_)),
		weightsFile_(Core::Configuration::config(paramWeightsFile_))
{}

void LibSvmToLogLinear::readData() {
	// read the training data
	Features::FeatureReader fr;
	fr.initialize();
	trainingData_.resize(fr.featureDimension(), fr.totalNumberOfFeatures());
	u32 idx = 0;
	while (fr.hasFeatures()) {
		const Math::Vector<Float>& f = fr.next();
		for (u32 d = 0; d < f.size(); d++)
			trainingData_.at(d, idx) = f.at(d);
		idx++;
	}
	// read probA and probB
	require(!probAFile_.empty());
	probA_.read(probAFile_);
	require(!probBFile_.empty());
	probB_.read(probBFile_);
	// read the bias
	require(!rhoFile_.empty());
	bias_.read(rhoFile_);
	bias_.scale(-1.0);
	require_eq(bias_.size(), probA_.size());
	require_eq(bias_.size(), probB_.size());
	// read the support vector coefficients
	require(!supportVectorCoefficientsFile_.empty());
	svCoef_.resize(bias_.size());
	std::ifstream f_svCoef(supportVectorCoefficientsFile_.c_str());
	std::string line;
	u32 c = 0;
	while (std::getline(f_svCoef, line)) {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, line);
		svCoef_.at(c).resize(tmp.size());
		for (u32 i = 0; i < tmp.size(); i++) {
			svCoef_.at(c).at(i) = atof(tmp.at(i).c_str());
		}
		c++;
	}
	f_svCoef.close();
	// read the support vector indices
	require(!supportVectorIndexFile_.empty());
	svIdx_.resize(bias_.size());
	std::ifstream f_svIdx(supportVectorIndexFile_.c_str());
	c = 0;
	while (std::getline(f_svIdx, line)) {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, line);
		svIdx_.at(c).resize(tmp.size());
		for (u32 i = 0; i < tmp.size(); i++) {
			svIdx_.at(c).at(i) = atoi(tmp.at(i).c_str());
		}
		c++;
	}
	f_svIdx.close();
	// ensure consistency of the files
	for (u32 i = 0; i < bias_.size(); i++) {
		require_eq(svCoef_.at(i).size(), svIdx_.at(i).size());
	}
}

void LibSvmToLogLinear::convert() {
	readData();
	weights_.resize(trainingData_.nRows(), bias_.size());
	weights_.setToZero();
	for (u32 c = 0; c < bias_.size(); c++) {
		// accumulate all support vectors for this class
		for (u32 sv = 0; sv < svIdx_.at(c).size(); sv++) {
#pragma omp parallel for
			for (u32 d = 0; d < weights_.nRows(); d++) {
				weights_.at(d, c) += trainingData_.at(d, svIdx_.at(c).at(sv)) * svCoef_.at(c).at(sv);
			}
		}
	}
	// prepare for use with sigmoid function
	weights_.multiplyColumnsByScalars(probA_);
	bias_.elementwiseMultiplication(probA_);
	bias_.add(probB_);
	// save result
	require(!biasFile_.empty());
	require(!weightsFile_.empty());
	bias_.write(biasFile_);
	weights_.write(weightsFile_);
}
