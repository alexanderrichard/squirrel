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
 * Statistics.hh
 *
 *  Created on: May 20, 2014
 *      Author: richard
 */

#ifndef NN_STATISTICS_HH_
#define NN_STATISTICS_HH_

#include <Core/CommonHeaders.hh>
#include "NeuralNetwork.hh"

namespace Nn {

class Statistics
{
public:
	// defines which statistics are accumulated
	// enumerate with 0,1,2,4,8,...
	enum StatisticTypes {
		none = 0,
		classificationStatistics = 1,
		objectiveFunctionStatistics = 2,
		gradientStatistics = 4,
		sequenceCount = 8
	};
private:
	bool needsClassificationStatistics_;
	bool needsObjectiveFunction_;
	bool needsGradient_;
	bool needsSequenceCount_;

	bool isNormalized_;
	bool isInitialized_;

	u32 nObservations_;
	u32 nSequences_; // for rnn training
	// classification statistics
	u32 nClassificationErrors_;
	// objective function statistics
	Float objectiveFunctionValue_;
	// gradient statistics
	std::vector<Vector> biasGradient_;
	std::vector<Matrix> weightsGradient_;

	std::map<std::string, u32> layerNameToIndex_;
	std::map<std::string, u32> connectionNameToIndex_;

	bool isComputing_;
public:
	Statistics(StatisticTypes types);
	virtual ~Statistics() {}
	void initialize(const NeuralNetwork& network);
	void reset();

	void saveStatistics(const std::string& filename);
	bool checkConsistency(Core::BinaryStream& s);
	void loadStatistics(const std::string& filename);

	bool needsClassificationStatistics() const { return needsClassificationStatistics_; }
	bool needsObjectiveFunction() const { return needsObjectiveFunction_; }
	bool needsGradient() const { return needsGradient_; }

	bool isNormalized() const { return isNormalized_; }

	void normalize();

	u32 nObservations();
	u32 nSequences();
	u32 nClassificationErrors();
	Float objectiveFunction();
	Vector& biasGradient(const std::string& layerName, u32 port);
	Matrix& weightsGradient(const std::string& connectionName);

	Float gradientNorm();

	void addToObjectiveFunction(Float value);
	void increaseNumberOfObservations(u32 nObservations);
	void increaseNumberOfSequences(u32 nSequences);
	void increaseNumberOfClassificationErrors(u32 nErrors);

	bool isComputing() const { return isComputing_; }
	void initComputation(bool sync = true);
	void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_STATISTICS_HH_ */
