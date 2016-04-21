#ifndef FEATURETRANSFORMATION_FRAMEPADDING_HH_
#define FEATURETRANSFORMATION_FRAMEPADDING_HH_

#include <Core/CommonHeaders.hh>
#include <Features/FeatureReader.hh>
#include <Features/FeatureWriter.hh>

namespace FeatureTransformation {

/*
 * class takes input sequences, pads zeros at the front or back to ensure a specific length for the input sequences
 * the result is written to a new feature cache
 */
class FramePadding
{
private:
	static const Core::ParameterInt paramFramesToPad_;
	static const Core::ParameterEnum paramPaddingPosition_;
	static const Core::ParameterEnum paramPaddingType_;
	enum PaddingPosition { front, back };
	enum PaddingType { zeros, copyFrame };

	u32 framesToPad_;
	PaddingPosition position_;
	PaddingType type_;

	Features::SequenceFeatureReader featureReader_;
	Features::SequenceFeatureWriter featureWriter_;
public:
	FramePadding();
	void pad();
};

} // namespace

#endif /* FEATURETRANSFORMATION_FRAMEPADDING_HH_ */
