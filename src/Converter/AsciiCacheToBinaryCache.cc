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

#include "AsciiCacheToBinaryCache.hh"

using namespace Converter;

const Core::ParameterString AsciiCacheToBinaryCache::paramInputFile_("ascii-cache-file", "", "converter.ascii-to-cache");

const Core::ParameterBool AsciiCacheToBinaryCache::paramIsLabelCache_("is-label-cache", false, "converter.ascii-to-cache");

AsciiCacheToBinaryCache::AsciiCacheToBinaryCache() :
		isLabelCache_(Core::Configuration::config(paramIsLabelCache_))
{}

void AsciiCacheToBinaryCache::convert() {
	std::string inputFile = Core::Configuration::config(paramInputFile_);
	require(!inputFile.empty());
	in_.open(inputFile, std::ios::in);

	/* read header */
	u32 featureType; in_ >> featureType;		// 0 for single, 1 for sequence
	u32 nFeatures; in_ >> nFeatures;			// total number of feature vectors in the cache
	u32 dimension; in_ >> dimension;			// feature vector dimension
	u32 nSequences = 0;
	if (featureType == 1) {					// in case of sequences:
		in_ >> nSequences;					// total number of sequences
	}

	if (isLabelCache_) {
		require_eq(dimension, 1);
	}

	if (featureType == 0)
		convertSingleFeatures(nFeatures, dimension);
	else
		convertSequenceFeatures(nSequences, dimension);
}

void AsciiCacheToBinaryCache::convertSingleFeatures(u32 nFeatures, u32 dimension) {
	if (!isLabelCache_) {
		Features::FeatureWriter featureWriter;
		Math::Vector<Float> v(dimension);
		for (u32 i = 0; i < nFeatures; i++) {
			for (u32 d = 0; d < dimension; d++) {
				in_ >> v.at(d);
			}
			featureWriter.write(v);
		}
		featureWriter.finalize();
	}
	else {
		Features::LabelWriter labelWriter;
		std::vector<u32> v(dimension);
		for (u32 i = 0; i < nFeatures; i++) {
			for (u32 d = 0; d < dimension; d++) {
				in_ >> v.at(d);
			}
			labelWriter.write(v);
		}
		labelWriter.finalize();
	}
}

void AsciiCacheToBinaryCache::convertSequenceFeatures(u32 nSequences, u32 dimension) {
	if (!isLabelCache_) {
		Features::SequenceFeatureWriter featureWriter;
		Math::Matrix<Float> m;
		std::vector<u32> timestamps;
		for (u32 s = 0; s < nSequences; s++) {
			u32 nFeatures; in_ >> nFeatures;
			m.resize(dimension, nFeatures);
			timestamps.clear();
			for (u32 i = 0; i < nFeatures; i++) {
				u32 timestamp; in_ >> timestamp;
				timestamps.push_back(timestamp);
				for (u32 d = 0; d < dimension; d++) {
					in_ >> m.at(d, i);
				}
			}
			featureWriter.write(timestamps, m);
		}
	}
	else {
		Features::SequenceLabelWriter labelWriter;
		std::vector<u32> labels;
		std::vector<u32> timestamps;
		for (u32 s = 0; s < nSequences; s++) {
			u32 nFeatures; in_ >> nFeatures;
			labels.clear();
			timestamps.clear();
			for (u32 i = 0; i < nFeatures; i++) {
				u32 timestamp; in_ >> timestamp;
				timestamps.push_back(timestamp);
				u32 label; in_ >> label;
				labels.push_back(label);
			}
			labelWriter.write(timestamps, labels);
		}
	}
}
