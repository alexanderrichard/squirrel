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

#ifndef NN_PORTFUSIONLAYER_HH_
#define NN_PORTFUSIONLAYER_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Layer.hh"

namespace Nn {

/*
 * PortFusionLayer
 *
 * base class to combine all input ports to a single output port
 */
class PortFusionLayer : public Layer
{
private:
	typedef Layer Precursor;
	MatrixContainer outputActivations_;
	MatrixContainer outputErrorSignals_;
public:
	PortFusionLayer(const char* name);
	virtual ~PortFusionLayer() {}
	virtual void initialize(u32 maxMemory);

	virtual u32 nOutputPorts() const { return 1; }

	virtual Matrix& activationsOut(u32 timeframe, u32 port);
	virtual Matrix& errorSignalOut(u32 timeframe, u32 port);

	// we need to override some of the standard layer functions...
	virtual void addTimeframe(u32 minibatchSize, bool initWithZero = false);
	virtual void addEmptyTimeframe();
	virtual void resizeTimeframe(u32 timeframe, u32 nRows, u32 nColumns);
	virtual void setMaximalMemory(u32 maxMemory);
	virtual void reset();

	virtual void setActivationVisibility(u32 timeframe, u32 nVisibleColumns);
	virtual void setErrorSignalVisibility(u32 timeframe, u32 nVisibleColumns);

	virtual void initComputation(bool sync = true);
	virtual void finishComputation(bool sync = true);
};

/*
 * MaxoutLayer
 */
class MaxoutLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
public:
	MaxoutLayer(const char* name);
	virtual ~MaxoutLayer() {}

	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

/*
 * GatedRecurrentUnitLayer
 */
class GatedRecurrentUnitLayer : public PortFusionLayer
{
private:
	typedef PortFusionLayer Precursor;
	static const Core::ParameterString paramOldGruFilename_;
	static const Core::ParameterString paramNewGruFilename_;
	std::vector<bool> blockedInputPorts_;
public:
	GatedRecurrentUnitLayer(const char* name);
	virtual ~GatedRecurrentUnitLayer() {}

	virtual void blockPorts();
	virtual void unblockPorts();
	virtual bool isInputPortBlocked(u32 port);

	virtual void addInternalConnections(NeuralNetwork& network);
	virtual u32 nInputPorts() const { return 4; }
	virtual void forward();
	virtual void backpropagate(u32 timeframe);
};

} // namespace

#endif /* NN_PORTFUSIONLAYER_HH_ */
