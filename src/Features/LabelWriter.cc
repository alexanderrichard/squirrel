#include "LabelWriter.hh"
#include "FeatureCache.hh"

using namespace Features;

/*
 * LabelWriter
 */

LabelWriter::LabelWriter(const char* name) :
		Precursor(name, FeatureCache::labelCache)
{}

void LabelWriter::write(const Math::Matrix<Float>& featureSequence) {
	Precursor::write(featureSequence);
}

void LabelWriter::write(const Math::Vector<Float>& feature) {
	Precursor::write(feature);
}

void LabelWriter::write(u32 label) {
	f32 label_f32 = reinterpret_cast<f32 &>(label);
	Math::Vector<Float> v(1);
	v.at(0) = label_f32;
	write(v);
}

void LabelWriter::write(const std::vector<u32>& labelSequence) {
	for (u32 i = 0; i < labelSequence.size(); i++) {
		write(labelSequence.at(i));
	}
}


/*
 * SequenceLabelWriter
 */

SequenceLabelWriter::SequenceLabelWriter(const char* name) :
		Precursor(name, FeatureCache::labelCache)
{}

void SequenceLabelWriter::write(std::vector<u32>& timestamps, const Math::Matrix<Float>& featureSequence) {
	Precursor::write(timestamps, featureSequence);
}

void SequenceLabelWriter::write(const Math::Matrix<Float>& featureSequence) {
	Precursor::write(featureSequence);
}

void SequenceLabelWriter::write(const std::vector<u32>& labelSequence) {
	Math::Matrix<Float> m(1, labelSequence.size());
	for (u32 i = 0; i < m.nColumns(); i++) {
		u32 label = labelSequence.at(i);
		f32 label_f32 = reinterpret_cast<f32 &>(label);
		m.at(0,i) = label_f32;
	}
	write(m);
}

void SequenceLabelWriter::write(std::vector<u32>& timestamps, const std::vector<u32>& labelSequence) {
	Math::Matrix<Float> m(1, labelSequence.size());
	for (u32 i = 0; i < m.nColumns(); i++) {
		u32 label = labelSequence.at(i);
		f32 label_f32 = reinterpret_cast<f32 &>(label);
		m.at(0,i) = label_f32;
	}
	write(timestamps, m);
}
