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

#include "DenseTrajectoriesConversion.hh"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>

using namespace Converter;

const Core::ParameterString DenseTrajectoriesConversion::paramTrajectoryFile_("trajectory-file", "",
		"dense-trajectory-conversion");

const Core::ParameterString DenseTrajectoriesConversion::paramCacheFileBasename_("feature-cache-basename", "",
		"dense-trajectory-conversion");

DenseTrajectoriesConversion::DenseTrajectoriesConversion() :
		trajectoryFile_(Core::Configuration::config(paramTrajectoryFile_)),
		cacheFileBasename_(Core::Configuration::config(paramCacheFileBasename_)),
		nFeatures_(0)
{}

void DenseTrajectoriesConversion::writeHeader(Core::BinaryStream& cache, u32 featureDim) {
    // write identifier (#c)
    cache << "#c";
    // write version (1)
    cache << (u8)1;
    // write feature type (1 for sequence)
    cache << (u8)1;
    // write number of feature vectors in cache (0 for the beginning, is changed later)
    cache << (u32)0;
    // write feature vector dimension
    cache << (u32)featureDim;
    // write number of sequences in cache
    cache << (u32)1;
}

void DenseTrajectoriesConversion::writeData(Core::BinaryStream& stream, u32 featureDim, f32* data) {
    // write the timestamp
    for (u32 i = 0; i < featureDim; i++) {
        stream << data[i];
    }
}

void DenseTrajectoriesConversion::convertToSequenceCache(std::string& filename) {
    Core::CompressedStream file(filename, std::ios::in);
    f32 data;
    std::vector<TrajContainer> sequence;
    // extract data
    while (!file.eof()) {
        TrajContainer tc;
        // read time stamp
        file >> tc.timestamp;
        // discard entries next 9 entries
        for (u32 i = 0; i < 9; i++)
            file >> data;
        // check again for eof since file ends with a tab (-> we would read an additional, invalid feature)
        if (file.eof()) {
            break;
        }
        // read traj
        for (u32 i = 0; i < 30; i++)
            file >> tc.traj[i];
        // read hog
        for (u32 i = 0; i < 96; i++)
            file >> tc.hog[i];
        // read hof
        for (u32 i = 0; i < 108; i++)
            file >> tc.hof[i];
        // read mbhx
        for (u32 i = 0; i < 96; i++)
            file >> tc.mbhx[i];
        // read mbhy
        for (u32 i = 0; i < 96; i++)
            file >> tc.mbhy[i];
        sequence.push_back(tc);
    }
    // write data to cache
    u32 sequenceLength = sequence.size();
    cacheTraj_ << sequenceLength;
    for (u32 l = 0; l < sequenceLength; l++) {
        cacheTraj_ << sequence.at(l).timestamp;
        writeData(cacheTraj_, 30, sequence.at(l).traj);
    }
    cacheHog_ << sequenceLength;
    for (u32 l = 0; l < sequenceLength; l++) {
        cacheHog_ << sequence.at(l).timestamp;
        writeData(cacheHog_, 96, sequence.at(l).hog);
    }
    cacheHof_ << sequenceLength;
    for (u32 l = 0; l < sequenceLength; l++) {
        cacheHof_ << sequence.at(l).timestamp;
        writeData(cacheHof_, 108, sequence.at(l).hof);
    }
    cacheMbhx_ << sequenceLength;
    for (u32 l = 0; l < sequenceLength; l++) {
        cacheMbhx_ << sequence.at(l).timestamp;
        writeData(cacheMbhx_, 96, sequence.at(l).mbhx);
    }
    cacheMbhy_ << sequenceLength;
    for (u32 l = 0; l < sequenceLength; l++) {
        cacheMbhy_ << sequence.at(l).timestamp;
        writeData(cacheMbhy_, 96, sequence.at(l).mbhy);
    }
    // update nFeatures and nSequences
    nFeatures_ += sequenceLength;
}

void DenseTrajectoriesConversion::openOutputCaches(std::string& filename) {
    std::string strTraj = filename + ".traj.cache";
    std::string strHog = filename + ".hog.cache";
    std::string strHof = filename + ".hof.cache";
    std::string strMbhx = filename + ".mbhx.cache";
    std::string strMbhy = filename + ".mbhy.cache";
    cacheTraj_.open(strTraj, std::ios::out);
    cacheHog_.open(strHog, std::ios::out);
    cacheHof_.open(strHof, std::ios::out);
    cacheMbhx_.open(strMbhx, std::ios::out);
    cacheMbhy_.open(strMbhy, std::ios::out);
    writeHeader(cacheTraj_, 30);
    writeHeader(cacheHog_, 96);
    writeHeader(cacheHof_, 108);
    writeHeader(cacheMbhx_, 96);
    writeHeader(cacheMbhy_, 96);
}

void DenseTrajectoriesConversion::closeOutputCaches() {
    cacheTraj_.close();
    cacheHog_.close();
    cacheHof_.close();
    cacheMbhx_.close(),
    cacheMbhy_.close();
}

void DenseTrajectoriesConversion::updateHeader(std::string& filename) {
    FILE* pf;
    // update number of features
    pf = fopen(filename.c_str(), "r+w");
    if (!pf) {
    	std::cerr << "DenseTrajectoriesConversion: Could not open file " << filename << ". Abort." << std::endl;
    	exit(1);
    }
    fseek(pf, 4, SEEK_SET); // skip first four bytes of header
    fwrite(&nFeatures_, sizeof(u32), 1, pf);
    fclose(pf);
    // update number of sequences
    pf = fopen(filename.c_str(), "r+w");
    if (!pf) {
    	std::cerr << "DenseTrajectoriesConversion: Could not open file " << filename << ". Abort." << std::endl;
    	exit(1);
    }
}

void DenseTrajectoriesConversion::convert() {
	openOutputCaches(cacheFileBasename_);
	convertToSequenceCache(trajectoryFile_);
	closeOutputCaches();

	std::string strTraj = cacheFileBasename_ + ".traj.cache";
	std::string strHog = cacheFileBasename_ + ".hog.cache";
	std::string strHof = cacheFileBasename_ + ".hof.cache";
	std::string strMbhx = cacheFileBasename_ + ".mbhx.cache";
	std::string strMbhy = cacheFileBasename_ + ".mbhy.cache";
	updateHeader(strTraj);
	updateHeader(strHog);
	updateHeader(strHof);
	updateHeader(strMbhx);
	updateHeader(strMbhy);
}
