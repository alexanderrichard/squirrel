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
 * Error.hh
 *
 *  Created on: May 20, 2016
 *      Author: richard
 */

#ifndef CORE_ERROR_HH_
#define CORE_ERROR_HH_

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "Types.hh"

namespace Core {

class Error
{
private:
	static Error* theInstance_;
	static Error& getInstance();
	Error() {}
public:
	static void abort(std::ostream& stream);
	static Error& msg(const char* msg = "");
	Error& operator<<(void (*fptr)(std::ostream&));
	Error& operator<<(u8 n);
	Error& operator<<(u32 n);
	Error& operator<<(u64 n);
	Error& operator<<(s8 n);
	Error& operator<<(s32 n);
	Error& operator<<(s64 n);
	Error& operator<<(f32 n);
	Error& operator<<(f64 n);
	Error& operator<<(bool n);
	Error& operator<<(char n);
	Error& operator<<(const char* n);
	Error& operator<<(const std::string& n);
};

} // namespace

#endif /* CORE_ERROR_HH_ */
