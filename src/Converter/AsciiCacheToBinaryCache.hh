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
