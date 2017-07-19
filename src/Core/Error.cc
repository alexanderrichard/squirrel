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
 * Error.cc
 *
 *  Created on: May 20, 2016
 *      Author: richard
 */

#include "Error.hh"
#include "Core/Utils.hh"

using namespace Core;

Error* Error::theInstance_ = 0;

Error& Error::getInstance() {
	if (theInstance_ == 0)
		theInstance_ = new Error;
	return *theInstance_;
}

void Error::abort(std::ostream& stream) {
	std::cerr << std::endl << "Abort." << std::endl;
	require(false);
}

Error& Error::msg(const char* msg) {
	std::cerr << std::endl << "ERROR:" << std::endl << msg;
	return getInstance();
}

Error& Error::operator<<(void (*fptr)(std::ostream&)) { fptr(std::cerr); return getInstance(); }

Error& Error::operator<<(u8 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(u32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(u64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s8 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(s64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(f32 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(f64 n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(bool n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(char n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(const char* n) { std::cerr << n; return getInstance(); }
Error& Error::operator<<(const std::string& n) { std::cerr << n; return getInstance(); }
