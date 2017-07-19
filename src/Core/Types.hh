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
 * Types.hh
 *
 *  Created on: 24.03.2014
 *      Author: richard
 */

#ifndef CORE_TYPES_HH_
#define CORE_TYPES_HH_

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned int u32;
typedef signed int s32;
typedef unsigned long int u64;
typedef signed long int s64;
typedef float f32;
typedef double f64;

// use this type to switch between f32 and f64
typedef f32 Float;

class Types
{
public:
	template<typename T>
	static const T min();

	template<typename T>
	static const T max();

	template<typename T>
	static const T absMin();

	template<typename T>
	static const bool isNan(T val);

	template<typename T>
	static const T inf();
};

#endif /* CORE_TYPES_HH_ */
