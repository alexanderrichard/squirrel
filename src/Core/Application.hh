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

#ifndef CORE_APPLICATION_HH_
#define CORE_APPLICATION_HH_

#define APPLICATION(A)                          \
	int main(int argc, const char* argv[]) {    \
		A app;                                  \
		app.run(argc, argv);                    \
		return 0;                               \
	}

// handler for segmentation faults
void handler(int sig);

namespace Core {

class Application
{
private:
public:
	virtual ~Application() {}
	virtual void run(int argc, const char* argv[]);
	virtual void main() {};
};

} // namespace

#endif /* CORE_APPLICATION_HH_ */
