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

#ifndef FEATURES_CACHECOMBINATION_HH_
#define FEATURES_CACHECOMBINATION_HH_

#include <Core/CommonHeaders.hh>
#include "FeatureReader.hh"
#include "FeatureWriter.hh"

namespace Features {

class BaseCacheCombination
{
private:
	static const Core::ParameterInt paramNumberOfCaches_;
	static const Core::ParameterFloatList paramCacheWeights_;
	static const Core::ParameterEnum paramCombinationMethod_;
protected:
	enum CombinationMethod { concatenate, maxPooling, averagePooling, sumPooling };

	u32 nCaches_;
	std::vector<f32> cacheWeights_;
	CombinationMethod method_;

	virtual void concat() = 0;
	virtual void pool() = 0;
public:
	BaseCacheCombination();
	virtual ~BaseCacheCombination() {};
	// has to be called before the cache combiner can be used
	virtual void initialize() = 0;
	/*
	 * combine the given feature caches to one cache using the combination method specified via paramCombinationMethod_
	 */
	virtual void combine();
	virtual void finalize() = 0;
};

/*
 * convert multiple non-sequence feature caches to one cache
 */
class CacheCombination : public BaseCacheCombination
{
private:
	typedef BaseCacheCombination Precursor;
protected:
	std::vector<FeatureReader*> featureReader_;
	FeatureWriter featureWriter_;

	virtual void concat();
	virtual void pool();
public:
	CacheCombination();
	virtual ~CacheCombination();
	virtual void initialize();
	virtual void finalize();
};

/*
 * convert multiple sequence feature caches to one cache
 */
class SequenceCacheCombination : public BaseCacheCombination
{
private:
	typedef BaseCacheCombination Precursor;
protected:
	std::vector<SequenceFeatureReader*> featureReader_;
	SequenceFeatureWriter featureWriter_;

	virtual void concat();
	virtual void pool();
public:
	SequenceCacheCombination();
	virtual ~SequenceCacheCombination();
	virtual void initialize();
	virtual void finalize();
};

} // namespace

#endif /* FEATURES_CACHECOMBINATION_HH_ */
