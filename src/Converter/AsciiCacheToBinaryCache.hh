#ifndef CONVERTER_ASCIICACHETOBINARYCACHE_HH_
#define CONVERTER_ASCIICACHETOBINARYCACHE_HH_

#include <Core/CommonHeaders.hh>
#include <Features/FeatureReader.hh>
#include <Features/FeatureWriter.hh>
#include <Features/LabelWriter.hh>

namespace Converter {

class AsciiCacheToBinaryCache
{
private:
	static const Core::ParameterString paramInputFile_;
	static const Core::ParameterBool paramIsLabelCache_;
private:
	Core::AsciiStream in_;
	bool isLabelCache_;

	void convertSingleFeatures(u32 nFeatures, u32 dimension);
	void convertSequenceFeatures(u32 nSequences, u32 dimension);
public:
	AsciiCacheToBinaryCache();
	~AsciiCacheToBinaryCache() {}

	/*
	 * convert an ascii file to a feature cache
	 */
	void convert();
};

} // namespace

#endif /* CONVERTER_ASCIICACHETOBINARYCACHE_HH_ */
