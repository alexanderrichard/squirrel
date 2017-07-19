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
 * Application.hh
 *
 *  Created on: Apr 10, 2014
 *      Author: richard
 */

#ifndef NN_APPLICATION_HH_
#define NN_APPLICATION_HH_

#include "Core/CommonHeaders.hh"
#include "Core/Application.hh"
#include "Types.hh"

namespace Nn {

class Application: public Core::Application
{
public:
	Application() {}
	virtual ~Application() {}
	void main();
};

class NeuralNetworkApplication {
private:
	static const Core::ParameterEnum paramAction_;
	static const Core::ParameterInt paramBatchSize_;
	enum Action { none, training, forwarding };
	u32 batchSize_;
private:
	void initialize();

public:
	NeuralNetworkApplication();
	virtual ~NeuralNetworkApplication() {}
	virtual void run();
};

} // namespace

#endif /* NN_APPLICATION_HH_ */
