#ifndef NN_ACTIVATIONLAYER_HH_
#define NN_ACTIVATIONLAYER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Layer.hh"

namespace Nn {

/*
 * SigmoidLayer
 */
class SigmoidLayer : public Layer
{
private:
	typedef Layer Precursor;
public:
	SigmoidLayer(const char* name);
	virtual ~SigmoidLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * TanhLayer
 */
class TanhLayer : public Layer
{
private:
	typedef Layer Precursor;
public:
	TanhLayer(const char* name);
	virtual ~TanhLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * SoftmaxLayer
 */
class SoftmaxLayer : public Layer
{
private:
	typedef Layer Precursor;
	static const Core::ParameterFloat paramEnhancementFactor_;
	Float enhancementFactor_;
public:
	SoftmaxLayer(const char* name);
	virtual ~SoftmaxLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
	virtual Float enhancementFactor() const { return enhancementFactor_; }
};

/*
 * Rectified Linear Units
 */
class RectifiedLayer : public Layer
{
private:
	typedef Layer Precursor;
public:
	RectifiedLayer(const char* name);
	virtual ~RectifiedLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * L2NormalizationLayer
 */
class L2NormalizationLayer : public Layer
{
private:
	typedef Layer Precursor;
	std::vector<MatrixContainer> normalization_;
public:
	L2NormalizationLayer(const char* name);
	virtual ~L2NormalizationLayer() {}
	virtual void initialize(u32 maxMemory = 1);
	virtual void addTimeframe(u32 minibatchSize, bool initWithZero = false);
	virtual void addEmptyTimeframe();
	virtual void setActivationVisibility(u32 timeframe, u32 nVisibleColumns);
	virtual void resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns);
	virtual void setMaximalMemory(u32 maxMemory);
	virtual void reset();
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * PowerNormalizationLayer
 */
class PowerNormalizationLayer : public Layer
{
private:
	typedef Layer Precursor;
	static const Core::ParameterFloat paramAlpha_;
	Float alpha_;
public:
	PowerNormalizationLayer(const char* name);
	virtual ~PowerNormalizationLayer() {}
	virtual void forward(u32 port);
	virtual void backpropagate(u32 timeframe, u32 port);
};

/*
 * SequenceNormalizationLayer
 *
 * normalizes the activations with the sequence lengths
 */
class SequenceLengthNormalizationLayer : public Layer
{
private:
	typedef Layer Precursor;
	Vector sequenceLengths_;
	std::vector<u32> u32SequenceLengths_;
public:
	SequenceLengthNormalizationLayer(const char* name);
	virtual ~SequenceLengthNormalizationLayer() {}
	virtual void forward(u32 port);
	virtual void finalizeForwarding();
	virtual void backpropagate(u32 timeframe, u32 port);
};

} // namespace

#endif /* NN_ACTIVATIONLAYER_HH_ */
