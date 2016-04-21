#ifndef ACTIONDETECTION_LENGTHMODELLING_HH_
#define ACTIONDETECTION_LENGTHMODELLING_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <map>

namespace ActionDetection {

/*
 * Base class for length models
 */
class LengthModel {
private:
	static const Core::ParameterEnum paramLengthModelType_;
	static const Core::ParameterInt paramNumberOfClasses_;
	static const Core::ParameterString paramTrainingFile_;
	enum LengthModelType { none, poissonModel, meanLengthModel };
protected:
	typedef std::vector<u32> TrainingData;
	u32 nClasses_;
	std::string trainingFile_;
	bool isInitialized_;
	// reads training data sorted by length in ascending order
	void readTrainingData(std::vector<TrainingData>& data);
public:
	LengthModel();
	virtual ~LengthModel() {}
	virtual void initialize() = 0;
	virtual void estimate() = 0;
	/*
	 * @return probability for length if class is c
	 */
	virtual Float probability(u32 length, u32 c);
	/*
	 * @return log-probability for length if class is c
	 */
	virtual Float logProbability(u32 length, u32 c) = 0;

	/*
	 * length model factory
	 * @return the created length model
	 */
	static LengthModel* create();
};

/*
 * class wise length model using Poisson distributions
 */
class PoissonModel : public LengthModel
{
private:
	static const Core::ParameterString paramLambdaVector_;
	typedef LengthModel Precursor;
protected:
	std::string lambdaFile_;
	Math::Vector<Float> lambdas_;
public:
	PoissonModel();
	virtual ~PoissonModel() {}
	virtual void initialize();
	virtual void estimate();
	/*
	 * @return log-probability for length if class is c
	 */
	virtual Float logProbability(u32 length, u32 c);
	/*
	 * @return lambda value for length model of class c
	 */
	Float lambda(u32 c);
};

/*
 * mean length model
 */
class MeanLengthModel : public LengthModel
{
private:
	static const Core::ParameterFloat paramMeanLength_;
	static const Core::ParameterFloat paramDecayFactor_;
	typedef LengthModel Precursor;
protected:
	Float meanLength_;
	Float decayFactor_;
public:
	MeanLengthModel();
	virtual ~MeanLengthModel() {}
	virtual void initialize();
	virtual void estimate() {}
	/*
	 * @return log-probability for length if class is c
	 */
	virtual Float logProbability(u32 length, u32 c);
};

}; // namespace

#endif /* ACTIONDETECTION_LENGTHMODELLING_HH_ */
