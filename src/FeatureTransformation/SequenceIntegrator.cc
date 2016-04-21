#include "SequenceIntegrator.hh"
#include <algorithm>
#include <cmath>

using namespace FeatureTransformation;

/*
 * SequenceIntegrator
 */
SequenceIntegrator::SequenceIntegrator()
{}

void SequenceIntegrator::integrate(Math::Matrix<Float>& sequence) {
	for (u32 t = 1; t < sequence.nColumns(); t++) {
#pragma omp parallel for
		for (u32 d = 0; d < sequence.nRows(); d++) {
			sequence.at(d, t) += sequence.at(d, t-1);
		}
	}
}

void SequenceIntegrator::integral(u32 t_start, u32 t_end,
		const Math::Matrix<Float>& sequence, Math::Vector<Float>& result) {

	// assume timestamps are ordered (t=0,1,2,...)
	require_le(t_start, t_end);
	require_lt(t_end, sequence.nColumns());
	result.resize(sequence.nRows());
#pragma omp parallel for
	for (u32 d = 0; d < result.nRows(); d++) {
		result.at(d) = sequence.at(d, t_end);
		if (t_start > 0)
			result.at(d) -= sequence.at(d, t_start-1);
	}
}

void SequenceIntegrator::integral(u32 t_start, u32 t_end,
		const Math::Matrix<Float>& sequence, const std::vector<u32>& timestamps,
		Math::Vector<Float>& result) {

	/* run through timestamps to search for first occurrence of t_start */
	u32 start_index = Types::max<u32>();
	for (u32 i = 0; i < timestamps.size(); i++) {
		if (timestamps.at(i) == t_start) {
			start_index = i;
			break;
		}
	}
	require_lt(start_index, timestamps.size());

	/* run backwards through timestamps to search for last occurrence of t_end */
	u32 end_index = Types::max<u32>();
	for (u32 i = timestamps.size(); i > 0; i--) {
		if (timestamps.at(i-1) == t_end) {
			end_index = i-1;
			break;
		}
	}
	require_lt(end_index, timestamps.size());

	/* get integral */
	integral(start_index, end_index, sequence, result);
}

/*
 * TemporallyLabeledSequenceSegmenter
 */
TemporallyLabeledSequenceSegmenter::TemporallyLabeledSequenceSegmenter() :
		Precursor()
{}

void TemporallyLabeledSequenceSegmenter::writeVector(const Math::Matrix<Float>& sequence, u32 label, u32 t_start, u32 t_end) {
	Math::Vector<Float> v;
	integral(t_start, t_end, sequence, v);
	featureWriter_.write(v);
	labelWriter_.write(label);
}

void TemporallyLabeledSequenceSegmenter::segment(const Math::Matrix<Float>& sequence, const std::vector<u32>& labels) {
	require_gt(labels.size(), 0);
	require_eq(labels.size(), sequence.nColumns());
	u32 t_start = 0;
	u32 t_end = 0;
	while (t_end < labels.size()) {
		while ((t_end < labels.size()) && (labels.at(t_start) == labels.at(t_end))) {
			t_end++;
		}
		t_end--;
		writeVector(sequence, labels.at(t_end), t_start, t_end);
		t_end++;
		t_start = t_end;
	}
}

void TemporallyLabeledSequenceSegmenter::segment() {
	featureReader_.initialize();
	while (featureReader_.hasSequences()) {
		const Math::Matrix<Float>& sequence = featureReader_.next();
		const std::vector<u32>& labels = featureReader_.labelSequence();
		segment(sequence, labels);
	}
}

/*
 * WindowedSequenceSegmenter
 */
const Core::ParameterInt WindowedSequenceSegmenter::paramWindowSize_("window-size", 1, "");

const Core::ParameterInt WindowedSequenceSegmenter::paramFrameSkip_("frame-skip", 1, "");

WindowedSequenceSegmenter::WindowedSequenceSegmenter() :
		Precursor(),
		windowSize_(Core::Configuration::config(paramWindowSize_)),
		frameSkip_(Core::Configuration::config(paramFrameSkip_))
{
	require_gt(windowSize_, 0);
	if (windowSize_ % 2 == 0) {
		windowSize_++;
		Core::Log::os("Window size is even. Increase to next odd value ") << windowSize_;
	}
	require_gt(frameSkip_, 0);
}

void WindowedSequenceSegmenter::segment() {
	featureReader_.initialize();
	while (featureReader_.hasSequences()) {
		// process the current sequence
		const Math::Matrix<Float>& sequence = featureReader_.next();
		u32 nSegments = std::ceil((Float)sequence.nColumns() / (Float)frameSkip_);
		Math::Matrix<Float> result(sequence.nRows(), nSegments);
		u32 segmentNo = 0;
		for (u32 t = 0; t < sequence.nColumns(); t += frameSkip_) {
			Math::Vector<Float> v;
			u32 t_start = std::max((s32)t - (s32)windowSize_ / 2, (s32)0);
			u32 t_end = std::min(t + windowSize_/2, sequence.nColumns()-1);
			integral(t_start, t_end, sequence, v);
			result.setColumn(segmentNo, v);
			segmentNo++;
		}
		featureWriter_.write(result);
	}
}
