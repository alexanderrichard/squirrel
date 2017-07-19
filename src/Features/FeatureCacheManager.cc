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

/*
 * FeatureCacheManager.cc
 *
 *  Created on: Apr 14, 2014
 *      Author: richard
 */

#include "FeatureCacheManager.hh"
#include <iostream>

using namespace Features;

/*
 * FeatureCachePrinter
 */
void FeatureCachePrinter::work() {
	FeatureReader reader;
	SequenceFeatureReader seqReader;
	FeatureCache::FeatureType type = FeatureCache::featureType(reader.getCacheFilename());

	switch (type) {

	case FeatureCache::vectors:
	case FeatureCache::images:
		reader.initialize();
		std::cout << std::endl;
		while (reader.hasFeatures()) {
			const Math::Vector<Float>& f = reader.next();
			std::cout << f.toString(true) << std::endl;
		}
		break;

	case FeatureCache::labels:
		reader.initialize();
		std::cout << std::endl;
		while (reader.hasFeatures()) {
			const Math::Vector<Float>& f = reader.next();
			std::cout << f.argAbsMax() << std::endl;
		}
		break;

	case FeatureCache::sequences:
	case FeatureCache::videos:
		seqReader.initialize();
		std::cout << std::endl;
		while (seqReader.hasSequences()) {
			const Math::Matrix<Float>& f = seqReader.next();
			std::cout << f.toString(true) << std::endl;
			std::cout << "#" << std::endl;
		}
		break;

	case FeatureCache::sequencelabels:
		seqReader.initialize();
		std::cout << std::endl;
		while (seqReader.hasSequences()) {
			const Math::Matrix<Float>& f = seqReader.next();
			Math::Vector<u32> labels(f.nColumns());
			f.argMax(labels);
			std::cout << labels.toString() << std::endl;
			std::cout << "#" << std::endl;
		}
		break;
	default:
		break; // this can not happen
	}
}
