#ifndef CONVERTER_LABELCONVERSION_HH_
#define CONVERTER_LABELCONVERSION_HH_

#include "Core/CommonHeaders.hh"
#include "Features/LabelWriter.hh"
#include "Features/FeatureReader.hh"
#include <string.h>

namespace Converter {

/*
 * convert a list of ascii label numbers (0,1,2,... line by line for each observation) to a label cache
 * in case of temporal label cache, lines have to be <timestamp> <label>, e.g.
 * 0 1
 * 1 3
 * 2 1
 * ...
 */
class AsciiLabelConverter
{
private:
	static const Core::ParameterString paramAsciiLabelFile_;
	static const Core::ParameterBool paramIsSequenceLabelFile_;
	std::string asciiLabelFile_;
	bool isSequenceLabelFile_;

	void _writeLabelCache();
	void _writeSequenceLabelCache();
public:
	AsciiLabelConverter();
	void writeLabelCache();
};

/*
 * convert a single-label cache to a sequence-label cache
 * expects the single-label cache and a sequence feature cache with the corresponding number of sequences as input
 */
class SingleLabelToSequenceLabelConverter
{
private:
	Features::SequenceFeatureReader featureReader_;
	Features::LabelReader labelReader_;
	Features::SequenceLabelWriter labelWriter_;
public:
	SingleLabelToSequenceLabelConverter();
	void convert();
};

} // namespace


#endif /* CONVERTER_LABELCONVERSION_HH_ */
