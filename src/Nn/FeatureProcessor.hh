#ifndef NN_LABELEDFEATUREPROCESSOR_HH_
#define NN_LABELEDFEATUREPROCESSOR_HH_

#include <Core/CommonHeaders.hh>
#include <Features/FeatureReader.hh>
#include <Features/LabeledFeatureReader.hh>
#include "Types.hh"
#include "Trainer.hh"

namespace Nn {

/*
 * base class for neural network feature processors
 */
class BaseFeatureProcessor
{
private:
	static const Core::ParameterEnum paramTrainingMode_;
	static const Core::ParameterInt paramBatchSize_;
	static const Core::ParameterInt paramMaxNumberOfEpochs_;
	static const Core::ParameterInt paramFirstEpoch_;
	static const Core::ParameterInt paramEpochLength_;
	static const Core::ParameterEnum paramFeatureType_;
	static const Core::ParameterInt paramSaveFrequency_;
public:
	enum FeatureType { single, sequence };
protected:
	enum TrainingMode { supervised, unsupervised };
protected:
	u32 batchSize_;
	u32 maxNumberOfEpochs_;
	u32 nProcessedEpochs_;
	u32 epochLength_;
	u32 nProcessedMinibatches_;
	u32 nProcessedObservations_;
	u32 dimension_;
	u32 totalNumberOfObservations_;

	u32 saveFrequency_;

	TrainingMode trainingMode_;

	Trainer* trainer_;

	bool isInitialized_;

	virtual void startNewEpoch() = 0;
public:
	BaseFeatureProcessor();
	virtual ~BaseFeatureProcessor();
	virtual void initialize() {}
	virtual void finalize();
	virtual bool isInitialized() const { return isInitialized_; }
	// are there unprocessed features?
	virtual bool hasUnprocessedFeatures() const = 0;
	// process a single batch
	virtual void processBatch() = 0;
	// process an epoch
	virtual void processEpoch();
	// process all epochs (maxNumberOfEpochs_ at the latest)
	virtual void processAllEpochs();

	/* factory */
	static BaseFeatureProcessor* createFeatureProcessor();
};

/*
 * feature processor for (un)labeled single features
 */
class FeatureProcessor : public BaseFeatureProcessor
{
private:
	typedef BaseFeatureProcessor Precursor;
private:
	Features::FeatureReader featureReader_;
	Features::LabeledFeatureReader labeledFeatureReader_;

	Matrix batch_;
	LabelVector labels_;

	virtual void startNewEpoch();
	void generateMinibatch();
public:
	FeatureProcessor();
	virtual ~FeatureProcessor() {}
	virtual void initialize();
	// are there unprocessed features?
	virtual bool hasUnprocessedFeatures() const;
	// process a single batch
	virtual void processBatch();
};

/*
 * feature processor for (un)labeled sequence features
 */
class SequenceFeatureProcessor : public BaseFeatureProcessor
{
private:
	typedef BaseFeatureProcessor Precursor;
private:
	Features::SequenceFeatureReader featureReader_;
	Features::LabeledSequenceFeatureReader labeledFeatureReader_;

	MatrixContainer batch_;
	LabelVector labels_;

	virtual void startNewEpoch();
	void generateMinibatch();
public:
	SequenceFeatureProcessor();
	virtual ~SequenceFeatureProcessor() {}
	virtual void initialize();
	// are there unprocessed features?
	virtual bool hasUnprocessedFeatures() const;
	// process a single batch
	virtual void processBatch();
};

} // namespace

#endif /* NN_LABELEDFEATUREPROCESSOR_HH_ */
