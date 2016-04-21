#include "LengthModelling.hh"
#include <Features/LabeledFeatureReader.hh>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <algorithm>

using namespace ActionDetection;

/*
 * LengthModel
 */
const Core::ParameterEnum LengthModel::paramLengthModelType_("type",
		"none, poisson-model, mean-length-model", "none", "length-model");

const Core::ParameterInt LengthModel::paramNumberOfClasses_("number-of-classes", 1, "length-model");

// training file: should be a file where in each line there are two numbers, the sequence length followed by the class index
const Core::ParameterString LengthModel::paramTrainingFile_("training-file", "", "length-model");

LengthModel::LengthModel() :
		nClasses_(Core::Configuration::config(paramNumberOfClasses_)),
		trainingFile_(Core::Configuration::config(paramTrainingFile_)),
		isInitialized_(false)
{
	require_gt(nClasses_, 0);
}

void LengthModel::readTrainingData(std::vector<TrainingData>& data) {
	if (trainingFile_.empty()) {
		std::cerr << "LengthModel::readTrainingData: no training-file given. Abort." << std::endl;
		exit(1);
	}
	data.clear();
	data.resize(nClasses_);

	std::ifstream train_in(trainingFile_.c_str());
	require(train_in.is_open());
	std::string line;

	while (std::getline(train_in, line)) {
		std::vector<std::string> v;
		Core::Utils::tokenizeString(v, line, " ");
		u32 sequenceLength = atoi(v[0].c_str());
		u32 label = atoi(v[1].c_str());
		require_lt(label, nClasses_);
		data.at(label).push_back(sequenceLength);
	}

	train_in.close();

	for (u32 c = 0; c < nClasses_; c++) {
		std::sort(data.at(c).begin(), data.at(c).end());
	}
}

Float LengthModel::probability(u32 length, u32 c) {
	require(isInitialized_);
	return std::exp(logProbability(length, c));
}

LengthModel* LengthModel::create() {
	LengthModel* p = 0;

	switch ((LengthModelType)Core::Configuration::config(paramLengthModelType_)) {
	case poissonModel:
		p = new PoissonModel();
		Core::Log::os("Created Poisson length model.");
		break;
	case meanLengthModel:
		p = new MeanLengthModel();
		Core::Log::os("Created mean length model.");
		break;
	case none:
	default:
		std::cerr << "Error: LengthModelling.cc: No length model type specified. Abort." << std::endl;
		exit(1);
	}

	return p;
}

/*
 * PoissonModel
 */
const Core::ParameterString PoissonModel::paramLambdaVector_("file", "", "length-model");

PoissonModel::PoissonModel() :
		Precursor(),
		lambdaFile_(Core::Configuration::config(paramLambdaVector_))
{}

void PoissonModel::initialize() {
	if (lambdaFile_.empty()) {
		std::cerr << "PoissonModel::initialize: no model file given. Abort." << std::endl;
		exit(1);
	}
	else {
		lambdas_.read(lambdaFile_);
		require_eq(lambdas_.size(), nClasses_);
	}
	isInitialized_ = true;
}

void PoissonModel::estimate() {
	Core::Log::openTag("length-model");
	Core::Log::os("Estimate length-model for ") << nClasses_ << " classes.";

	std::vector<TrainingData> data;
	readTrainingData(data);

	lambdas_.resize(nClasses_);
	lambdas_.setToZero();

	for (u32 c = 0; c < nClasses_; c++) {
		for (u32 i = 0; i < data.at(c).size(); i++) {
			lambdas_.at(c) += data.at(c).at(i);
		}
		lambdas_.at(c) /= data.at(c).size();
	}

	if (lambdaFile_.empty()) {
		Core::Log::os("No length-model file given. Do not save estimates.");
	}
	else {
		Core::Log::os("Save length-model to ") << lambdaFile_;
		lambdas_.write(lambdaFile_);
	}

	Core::Log::closeTag();

	isInitialized_ = true;
}

Float PoissonModel::logProbability(u32 length, u32 c) {
	require(isInitialized_);
	Float logFak = 0;
	for (u32 k = 2; k <= length; k++)
		logFak += std::log(k);
	return length * std::log(lambdas_.at(c)) - lambdas_.at(c) - logFak;
}

Float PoissonModel::lambda(u32 c) {
	require(isInitialized_);
	return lambdas_.at(c);
}

/*
 * MeanLengthModel
 */
const Core::ParameterFloat MeanLengthModel::paramMeanLength_("mean-length", 200, "length-model");

const Core::ParameterFloat MeanLengthModel::paramDecayFactor_("decay-factor", 0.9, "length-model");

MeanLengthModel::MeanLengthModel() :
		Precursor(),
		meanLength_(Core::Configuration::config(paramMeanLength_)),
		decayFactor_(Core::Configuration::config(paramDecayFactor_))
{}

void MeanLengthModel::initialize() {
	isInitialized_ = true;
}

Float MeanLengthModel::logProbability(u32 length, u32 c) {
	Float score = 0;
	if (length > meanLength_)
		score = std::log(decayFactor_) * (length - meanLength_);
	return score;
}
