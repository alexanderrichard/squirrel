#include "LearningRateSchedule.hh"

using namespace Nn;

/*
 * LearningRateSchedule
 */
const Core::ParameterEnum LearningRateSchedule::paramLearningRateSchedule_("method", "none, online-newbob", "none", "learning-rate-schedule");

const Core::ParameterFloat LearningRateSchedule::paramInitialLearningRate_("initial-learning-rate", 1.0, "learning-rate-schedule");

LearningRateSchedule::LearningRateSchedule() :
		learningRate_(Core::Configuration::config(paramInitialLearningRate_))
{}

void LearningRateSchedule::initialize() {
	Core::Log::os("initial learning rate: ") << learningRate_;
}

/* factory */
LearningRateSchedule* LearningRateSchedule::createLearningRateSchedule() {
	LearningRateSchedule* schedule = 0;
	switch ( (LearningRateScheduleType) Core::Configuration::config(paramLearningRateSchedule_) ) {
	case onlineNewbob:
		Core::Log::os("Create online-newbob learning rate scheduler.");
		schedule = new OnlineNewbob();
		break;
	case none:
	default:
		schedule = new LearningRateSchedule();
	}
	return schedule;
}

/*
 * Newbob
 */
const Core::ParameterFloat OnlineNewbob::paramLearningRateReductionFactor_("learning-rate-reduction-factor", 0.5, "learning-rate-schedule");

OnlineNewbob::OnlineNewbob() :
		Precursor(),
		factor_(Core::Configuration::config(paramLearningRateReductionFactor_)),
		nClassificationErrors_(0),
		nObservations_(0),
		oldObjectiveFunction_(Types::max<Float>()),
		objectiveFunction_(0),
		oldEpoch_(0),
		error_(1),
		isFirstEpoch_(true)
{}

void OnlineNewbob::updateLearningRate(Statistics& statistics, u32 epoch) {
	// keep the initial learning rate during the first epoch
	if (isFirstEpoch_) {
		isFirstEpoch_ = false;
		oldEpoch_ = epoch;
	}

	// update the learning rate if a new epoch has begun
	if (epoch > oldEpoch_) {
		oldEpoch_ = epoch;
		Float newError = (Float)nClassificationErrors_ / nObservations_;
		Core::Log::openTag("online-newbob");
		Core::Log::os("average classification error: ") << newError;
		Core::Log::os("average objective function value: ") << objectiveFunction_ / nObservations_;
		Core::Log::closeTag();
		// if error did not improve
		if (newError > error_) {
			learningRate_ *= factor_;
			Core::Log::os("online-newbob evaluation: ") << newError << " (new) vs. " << error_ << " (old)";
			Core::Log::os("set learning rate to ") << learningRate_;
		}
		// else if error did not change check objective function value
		else if (newError == error_) {
			if (objectiveFunction_ >= oldObjectiveFunction_) {
				learningRate_ *= factor_;
				Core::Log::os("online-newbob evaluation: ") << newError << " (new) vs. " << error_ << " (old), "
						<< objectiveFunction_ << " (new) vs. " << oldObjectiveFunction_ << " (old)";
				Core::Log::os("set learning rate to ") << learningRate_;
			}
		}
		error_ = newError;
		oldObjectiveFunction_ = objectiveFunction_;
		nObservations_ = 0;
		nClassificationErrors_ = 0;
		objectiveFunction_ = 0;
	}

	// accumulate the objective function
	nObservations_ += statistics.nObservations();
	nClassificationErrors_ += statistics.nClassificationErrors();
	objectiveFunction_ += statistics.objectiveFunction() * statistics.nObservations();
}
