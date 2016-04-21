#ifndef NN_LEARNINGRATESCHEDULE_HH_
#define NN_LEARNINGRATESCHEDULE_HH_

#include <Core/CommonHeaders.hh>
#include "Statistics.hh"

namespace Nn {

/*
 * base class for learning rate schedules
 */
class LearningRateSchedule
{
private:
	static const Core::ParameterEnum paramLearningRateSchedule_;
	static const Core::ParameterFloat paramInitialLearningRate_;
	enum LearningRateScheduleType { none, onlineNewbob };
protected:
	Float learningRate_;
public:
	LearningRateSchedule();
	virtual ~LearningRateSchedule() {}
	virtual void initialize();

	// return new learning rate
	// depending on the schedule, not all parameters may be relevant
	// however, a common interface is necessary
	virtual void updateLearningRate(Statistics& statistics, u32 epoch) {}
	virtual Float learningRate() { return learningRate_; }

	/* factory */
	static LearningRateSchedule* createLearningRateSchedule();
};

/*
 * online newbob learning rate schedule
 * compute avg classification error over one epoch
 * if the avg classification error did not decrease between two epochs, learning rate is halved
 */
class OnlineNewbob : public LearningRateSchedule
{
private:
	typedef LearningRateSchedule Precursor;
	static const Core::ParameterFloat paramLearningRateReductionFactor_;
private:
	Float factor_;
	u32 nClassificationErrors_;
	u32 nObservations_;
	Float oldObjectiveFunction_;
	Float objectiveFunction_;
	u32 oldEpoch_;
	Float error_;
	bool isFirstEpoch_;
public:
	OnlineNewbob();
	virtual ~OnlineNewbob() {}
	/*
	 * @param statistics contain information of classification error rate and objective function
	 * @param epoch the current epoch number
	 */
	virtual void updateLearningRate(Statistics& statistics, u32 epoch);
};

} // namespace

#endif /* NN_LEARNINGRATESCHEDULE_HH_ */
