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

#ifndef NN_STATISTICS_HH_
#define NN_STATISTICS_HH_

#include <Core/CommonHeaders.hh>
#include "NeuralNetwork.hh"

namespace Nn {

/*
 * statistics object contains various statistics, e.g.
 * number of accumulated observations, objective function value, gradient, ...
 */
class Statistics
{
public:
	// defines which statistics are accumulated
	// enumerate with 0,1,2,4,8,...
	enum StatisticTypes {
		none = 0,
		baseStatistics = 1, // accumulate number of observations, objective function value, ...
		gradientStatistics = 2 // accumulate the gradient
	};
private:
	bool needsBaseStatistics_;
	bool needsGradient_;

	bool isNormalized_;
	bool isInitialized_;

	u32 nObservations_;
	Float objectiveFunctionValue_;
	u32 nClassificationErrors_;

	std::map<std::string, u32> layerNameToIndex_;
	std::map<std::string, u32> connectionNameToIndex_;

	std::vector<Vector> biasGradient_;
	std::vector<Matrix> weightsGradient_;

	bool isComputing_;
public:
	Statistics(StatisticTypes types);
	virtual ~Statistics() {}
	/*
	 * @param network is used to determine for which weights/bias parameters the gradient has to be stored
	 */
	void initialize(const NeuralNetwork& network);
	/*
	 * reset the accumulated statistics
	 */
	void reset();

	/*
	 * save and load from/to file, check header for consistency
	 */
	void saveStatistics(const std::string& filename);
	bool checkConsistency(Core::BinaryStream& s);
	void loadStatistics(const std::string& filename);

	// return if the type of statistics is required
	bool needsBaseStatistics() const { return needsBaseStatistics_; }
	bool needsGradient() const { return needsGradient_; }

	/*
	 * @return true if the statistics are already normalized by the number of observations
	 */
	bool isNormalized() const { return isNormalized_; }

	/*
	 * normalize the accumulated statistics (divide by number of observations)
	 */
	void normalize();

	/*
	 * return accumulated statistics
	 */
	u32 nObservations();
	Float objectiveFunction();
	u32 nClassificationErrors();
	/*
	 * @param layerName name of the layer this bias gradient belongs to
	 * @param port the port the bias gradient belongs to
	 * @return reference to the bias gradient vector
	 */
	Vector& biasGradient(const std::string& layerName, u32 port);
	/*
	 * @param connectionName the name of the connection the weights gradient belongs to
	 * @return reference to the weight gradient matrix
	 */
	Matrix& weightsGradient(const std::string& connectionName);

	/*
	 * @return l1-norm of the gradient
	 */
	Float gradientNorm();

	/*
	 * accumulate base statistics
	 */
	void addToObjectiveFunction(Float value);
	void increaseNumberOfObservations(u32 nObservations);
	void increaseNumberOfClassificationErrors(u32 nErrors);

	bool isComputing() const { return isComputing_; }
	void initComputation(bool sync = true);
	void finishComputation(bool sync = true);
};

} // namespace

#endif /* NN_STATISTICS_HH_ */
