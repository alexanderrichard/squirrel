/*
 * Copyright 2016 Alexander Richard
 *
 * This file is part of Squirrel.
 *
 * Licensed under the Academic Free License 3.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should have received a copy of the License along with Squirrel.
 * If not, see <https://opensource.org/licenses/AFL-3.0>.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef FEATURETRANSFORMATION_SEQUENCEINTEGRATOR_HH_
#define FEATURETRANSFORMATION_SEQUENCEINTEGRATOR_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Matrix.hh>
#include <Features/LabeledFeatureReader.hh>
#include <Features/FeatureWriter.hh>
#include <Features/LabelWriter.hh>

namespace FeatureTransformation {

/*
 * SequenceIntegrator
 * compute integral sequences, i.e. for a feature vector f(t) at time t, the integral value is sum_k=0^t f(k)
 */
class SequenceIntegrator
{
public:
	SequenceIntegrator();
	/*
	 * @param sequence the sequence to be integrated
	 * @return the integral sequence in the parameter sequence
	 */
	void integrate(Math::Matrix<Float>& sequence);
	/*
	 * @return in the vector result the integral from t_start to t_end obtained from the integral sequence sequence
	 * timestamps are considered to be t=0,1,2,...
	 */
	void integral(u32 t_start, u32 t_end, const Math::Matrix<Float>& sequence, Math::Vector<Float>& result);
	/*
	 * @return in the vector result the integral from t_start to t_end obtained from the integral sequence sequence
	 * timestamps are specified in the timestamps vector, so there can be gaps (e.g. t=1,2,5,6,7...)
	 */
	void integral(u32 t_start, u32 t_end, const Math::Matrix<Float>& sequence, const std::vector<u32>& timestamps, Math::Vector<Float>& result);
};

/*
 * TemporallyLabeledSequenceSegmenter
 * computes integral segments of a temporally labeled sequence (feature cache is expected to contain integral sequences)
 * writes result to a feature cache and the corresponding labels to a label cache
 */
class TemporallyLabeledSequenceSegmenter : private SequenceIntegrator
{
private:
	typedef SequenceIntegrator Precursor;
	Math::Vector<u32> maxClassLengths_;
	Features::TemporallyLabeledSequenceFeatureReader featureReader_;
	Features::FeatureWriter featureWriter_;
	Features::LabelWriter labelWriter_;

	void writeVector(const Math::Matrix<Float>& sequence, u32 label, u32 t_start, u32 t_end);
	void segment(const Math::Matrix<Float>& sequence, const std::vector<u32>& labels);
public:
	TemporallyLabeledSequenceSegmenter();
	void segment();
};

/*
 * WindowedSequenceSegmenter
 * computes integral segments over windows of a sequence (feature cache is expected to contain integral sequences)
 * writes result to a feature cache and the corresponding labels to a label cache
 */
class WindowedSequenceSegmenter : private SequenceIntegrator
{
private:
	static const Core::ParameterInt paramWindowSize_;
	static const Core::ParameterInt paramFrameSkip_;
	typedef SequenceIntegrator Precursor;
	u32 windowSize_;
	u32 frameSkip_;
	Features::SequenceFeatureReader featureReader_;
	Features::SequenceFeatureWriter featureWriter_;
public:
	WindowedSequenceSegmenter();
	void segment();
};

} // namespace

#endif /* FEATURETRANSFORMATION_SEQUENCEINTEGRATOR_HH_ */
