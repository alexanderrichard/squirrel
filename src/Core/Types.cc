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

#include "Types.hh"
#include <limits>

template<typename T>
const T Types::absMin() {
	return 0;
}

template<>
const u8 Types::min<u8>() {
	return 0;
}

template<>
const u8 Types::max<u8>() {
	return std::numeric_limits<u8>::max();
}

template<>
const s8 Types::min<s8>() {
	return std::numeric_limits<s8>::min();
}

template<>
const s8 Types::max<s8>() {
	return std::numeric_limits<s8>::max();
}

template<>
const u32 Types::min<u32>() {
	return 0;
}

template<>
const u32 Types::max<u32>() {
	return std::numeric_limits<u32>::max();
}

template<>
const s32 Types::min<s32>() {
	return std::numeric_limits<s32>::min();
}

template<>
const s32 Types::max<s32>() {
	return std::numeric_limits<s32>::max();
}

template<>
const u64 Types::min<u64>() {
	return 0;
}

template<>
const u64 Types::max<u64>() {
	return std::numeric_limits<u64>::max();
}

template<>
const s64 Types::min<s64>() {
	return std::numeric_limits<s64>::min();
}

template<>
const s64 Types::max<s64>() {
	return std::numeric_limits<s64>::max();
}

template<>
const f32 Types::min<f32>() {
	return -std::numeric_limits<f32>::max();
}

template<>
const f32 Types::max<f32>() {
	return std::numeric_limits<f32>::max();
}

template<>
const f32 Types::absMin<f32>() {
	return std::numeric_limits<f32>::min();
}

template<>
const f64 Types::min<f64>() {
	return -std::numeric_limits<f64>::max();
}

template<>
const f64 Types::max<f64>() {
	return std::numeric_limits<f64>::max();
}

template<>
const f64 Types::absMin<f64>() {
	return std::numeric_limits<f64>::min();
}

template<>
bool const Types::isNan<f32>(f32 val) {
	union { f32 f; u32 x; } u = { val };
	return (u.x << 1) > 0xff000000u;
}

template<>
bool const Types::isNan<f64>(f64 val) {
	union { f64 f; u64 x; } u = { val };
	return (u.x << 1) > 0x7ff0000000000000u;
}

template<>
const f32 Types::inf<f32>() {
	return std::numeric_limits<f32>::infinity();
}

template<>
const f64 Types::inf<f64>() {
	return std::numeric_limits<f32>::infinity();
}
