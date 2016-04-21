#ifndef ACTIONDETECTION_SCORER_HH_
#define ACTIONDETECTION_SCORER_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Matrix.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Nn/NeuralNetwork.hh>

namespace ActionDetection {

/*
 * base class for action scorer
 */
class Scorer
{
private:
	static const Core::ParameterEnum paramScorerType_;
	enum ScorerType { none, neuralNetworkScorer };
protected:
	Math::Matrix<Float> const * sequence_;
	bool hasSequence_;
	bool isInitialized_;
public:
	Scorer();
	virtual ~Scorer() {}
	virtual void initialize();
	/*
	 * @param sequence sequence to work on
	 */
	void setSequence(const Math::Matrix<Float>& sequence);
	/*
	 * @param c the class index for the desired action class
	 * @param t_start specifies the hypothesized starting time of the current action
	 * @param t_end specifies the hypothesized ending time of the current action
	 * @return score (in log-domain) of class c at [t_start, t_end]
	 */
	virtual Float score(u32 c, u32 t_start, u32 t_end) = 0;
	/*
	 * @return number of classes
	 */
	virtual u32 nClasses() const = 0;

	/*
	 * factory
	 * @return the created scorer
	 */
	static Scorer* create();
};

/*
 * NeuralNetworkScorer
 * input sequence is expected to be an "integral image" over the features from t = 1,...,T
 */
class NeuralNetworkScorer : public Scorer
{
private:
	typedef Scorer Precursor;
	static const Core::ParameterString paramPriorFile_;
	static const Core::ParameterFloat paramPriorScale_;
	static const Core::ParameterInt paramBatchSize_;
	Nn::NeuralNetwork network_;
	std::string priorFile_;
	Float priorScale_;
	u32 batchSize_;
	Math::Vector<Float> prior_;
	Math::CudaMatrix<Float> networkInput_;
	u32 t_start_; // start frame for which scores have been precomputed
	u32 t_end_; // end frame for which scores have been precomputed

	virtual void generateVector(u32 t_start, u32 t_end, u32 column, Math::CudaMatrix<Float>& result);
public:
	NeuralNetworkScorer();
	virtual ~NeuralNetworkScorer() {}
	virtual void initialize();
	virtual void setSequence(const Math::Matrix<Float>& sequence);
	virtual Float score(u32 c, u32 t_start, u32 t_end);
	virtual u32 nClasses() const;
};

} // namespace

#endif /* ACTIONDETECTION_SCORER_HH_ */
