#include "FramePadding.hh"
#include <algorithm>
#include <functional>

using namespace FeatureTransformation;

const Core::ParameterInt FramePadding::paramFramesToPad_("frames-to-pad", 0, "frame-padding");

const Core::ParameterEnum FramePadding::paramPaddingPosition_("position", "front, back", "front", "frame-padding");

const Core::ParameterEnum FramePadding::paramPaddingType_("type", "zeros, copy-frame", "zeros", "frame-padding");

FramePadding::FramePadding() :
		framesToPad_(Core::Configuration::config(paramFramesToPad_)),
		position_((PaddingPosition) Core::Configuration::config(paramPaddingPosition_)),
		type_((PaddingType) Core::Configuration::config(paramPaddingType_))
{
	require_ge(framesToPad_, 0);
}

void FramePadding::pad() {
	featureReader_.initialize();

	while (featureReader_.hasSequences()) {
		const Math::Matrix<Float>& seq_in = featureReader_.next();
		Math::Matrix<Float> seq_out(seq_in.nRows(), seq_in.nColumns() + framesToPad_);

		// copy seq_in to seq_out depending on where to pad frames
		u32 copyColStart = (position_ == front) ? framesToPad_ : 0;
		if (!seq_out.needsU64Space()) {
			seq_out.copyBlockFromMatrix(seq_in, 0, 0, 0, copyColStart, seq_in.nRows(), seq_in.nColumns());
		}
		else {
			for (u32 i = 0; i < seq_in.nRows(); i++) {
				for (u32 j = 0; j < seq_in.nColumns(); j++) {
					seq_out.at(i, j + copyColStart) = seq_in.at(i, j);
				}
			}
		}

		// pad the frames
		u32 paddingStart = (position_ == front) ? 0 : seq_in.nColumns();
		for (u32 i = 0; i < framesToPad_; i++) {
			for (u32 d = 0; d < seq_in.nRows(); d++) {
				if (type_ == copyFrame)
					seq_out.at(d, i + paddingStart) = seq_in.at(d, (position_ == front) ? 0 : seq_in.nColumns() - 1);
				else
					seq_out.at(d, i + paddingStart) = 0;
			}
		}

		// adjust timestamps
		std::vector<u32> timestamps(featureReader_.currentTimestamps());
		// if padding at the front, maybe old timestamps have to be increased
		if (position_ == front) {
			u32 offset = (timestamps.front() < framesToPad_) ? framesToPad_ - timestamps.front() : 0;
			if (offset > 0) {
				std::transform(timestamps.begin(), timestamps.end(), timestamps.begin(),
						std::bind2nd(std::plus<u32>(), offset));
			}
			u32 t = timestamps.front();
			for (u32 i = 0; i < framesToPad_; i++) {
				t--;
				timestamps.insert(timestamps.begin(), t);
			}
		}
		// if padding at the end, additional timestamps can just be added
		else {
			u32 t = timestamps.back();
			for (u32 i = 0; i< framesToPad_; i++) {
				t++;
				timestamps.push_back(t);
			}
		}

		// write sequence
		featureWriter_.write(timestamps, seq_out);
	}
}
