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
 * Parameter.hh
 *
 *  Created on: 24.03.2014
 *      Author: richard
 */

#ifndef CORE_PARAMETER_HH_
#define CORE_PARAMETER_HH_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "Types.hh"

namespace Core {

class Parameter
{
private:
	const std::string name_;
	const std::string prefix_;
public:
	Parameter(const char* name, const char* prefix="");
	const std::string& getName() const;
	const std::string& getPrefix() const;
};

class ParameterInt : public Parameter
{
private:
	typedef Parameter Precursor;
	const s32 default_;
public:
	ParameterInt(const char* name, const s32 value = 0, const char* prefix = "");
	s32 getDefault() const;
};

class ParameterFloat : public Parameter
{
private:
	typedef Parameter Precursor;
	const f32 default_;
public:
	ParameterFloat(const char* name, const f32 value = 0.0, const char* prefix = "");
	f32 getDefault() const;
};

class ParameterChar : public Parameter
{
private:
	typedef Parameter Precursor;
	const char default_;
public:
	ParameterChar(const char* name, const char value = 0, const char* prefix = "");
	char getDefault() const;
};

class ParameterBool : public Parameter
{
private:
	typedef Parameter Precursor;
	const bool default_;
public:
	ParameterBool(const char* name, const bool value = true, const char* prefix = "");
	bool getDefault() const;
};

class ParameterString : public Parameter
{
private:
	typedef Parameter Precursor;
	const std::string default_;
public:
	ParameterString(const char* name, const char* value = "", const char* prefix = "");
	const std::string& getDefault() const;
};

class ParameterIntList : public Parameter
{
private:
	typedef Parameter Precursor;
	std::vector<s32> default_;
public:
	ParameterIntList(const char* name, const char* value = "", const char* prefix = "");
	const std::vector<s32>& getDefault() const;
};

class ParameterFloatList : public Parameter
{
private:
	typedef Parameter Precursor;
	std::vector<f32> default_;
public:
	ParameterFloatList(const char* name, const char* value = "", const char* prefix = "");
	const std::vector<f32>& getDefault() const;
};

class ParameterCharList : public Parameter
{
private:
	typedef Parameter Precursor;
	std::vector<char> default_;
public:
	ParameterCharList(const char* name, const char* value = "", const char* prefix = "");
	const std::vector<char>& getDefault() const;
};

class ParameterBoolList : public Parameter
{
private:
	typedef Parameter Precursor;
	std::vector<bool> default_;
public:
	ParameterBoolList(const char* name, const char* value = "", const char* prefix = "");
	const std::vector<bool>& getDefault() const;
};

class ParameterStringList : public Parameter
{
private:
	typedef Parameter Precursor;
	std::vector<std::string> default_;
public:
	ParameterStringList(const char* name, const char* value = "", const char* prefix = "");
	const std::vector<std::string>& getDefault() const;
};

class ParameterEnum : public Parameter
{
private:
	typedef Parameter Precursor;
	std::map<std::string, u32> enum_;
	u32 default_;
public:
	ParameterEnum(const char* name, const char* enumValues, const char* value, const char* prefix = "");
	const u32 getDefault() const;
	u32 num(const std::string& str) const;
	u32 num(const char* str) const;
};

} // namespace


#endif /* CORE_PARAMETER_HH_ */
