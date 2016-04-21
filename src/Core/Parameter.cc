#include "Parameter.hh"
#include "Utils.hh"
#include <stdlib.h>
#include <iostream>

using namespace Core;

Parameter::Parameter(const char* name, const char* prefix) :
		name_(std::string(name)),
		prefix_(std::string(prefix))
{}

const std::string& Parameter::getName() const {
	return name_;
}

const std::string& Parameter::getPrefix() const {
	return prefix_;
}

ParameterInt::ParameterInt(const char* name, const s32 value, const char* prefix) :
		Precursor(name, prefix),
		default_(value)
{}

s32 ParameterInt::getDefault() const {
	return default_;
}

ParameterFloat::ParameterFloat(const char* name, const f32 value, const char* prefix) :
		Precursor(name, prefix),
		default_(value)
{}

f32 ParameterFloat::getDefault() const {
	return default_;
}

ParameterChar::ParameterChar(const char* name, const char value, const char* prefix) :
		Precursor(name, prefix),
		default_(value)
{}

char ParameterChar::getDefault() const {
	return default_;
}

ParameterBool::ParameterBool(const char* name, const bool value, const char* prefix) :
		Precursor(name, prefix),
		default_(value)
{}

bool ParameterBool::getDefault() const {
	return default_;
}

ParameterString::ParameterString(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix),
		default_(std::string(value))
{}

const std::string& ParameterString::getDefault() const {
	return default_;
}

ParameterIntList::ParameterIntList(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(value);
	std::vector<std::string> v;
	Core::Utils::tokenizeString(v, tmp, ",");
	for (u32 i = 0; i < v.size(); i++) {
		default_.push_back(atoi(v[i].c_str()));
	}
}

const std::vector<s32>& ParameterIntList::getDefault() const {
	return default_;
}

ParameterFloatList::ParameterFloatList(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(value);
	std::vector<std::string> v;
	Core::Utils::tokenizeString(v, tmp, ",");
	for (u32 i = 0; i < v.size(); i++) {
		default_.push_back(atof(v[i].c_str()));
	}
}

const std::vector<f32>& ParameterFloatList::getDefault() const {
	return default_;
}

ParameterCharList::ParameterCharList(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(value);
	std::vector<std::string> v;
	Core::Utils::tokenizeString(v, tmp, ",");
	for (u32 i = 0; i < v.size(); i++) {
		default_.push_back(v[i].c_str()[0]);
	}
}

const std::vector<char>& ParameterCharList::getDefault() const {
	return default_;
}

ParameterBoolList::ParameterBoolList(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(value);
	std::vector<std::string> v;
	Core::Utils::tokenizeString(v, tmp, ",");
	for (u32 i = 0; i < v.size(); i++) {
		if ((v[i].compare(std::string("true")) == 0) ||
			(v[i].compare(std::string("TRUE")) == 0) ||
			(v[i].compare(std::string("yes")) == 0) ||
			(v[i].compare(std::string("1")) == 0)) {
			default_.push_back(true);
		}
		else {
			default_.push_back(false);
		}
	}
}

const std::vector<bool>& ParameterBoolList::getDefault() const {
	return default_;
}

ParameterStringList::ParameterStringList(const char* name, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(value);
	Core::Utils::tokenizeString(default_, tmp, ", ");
}

const std::vector<std::string>& ParameterStringList::getDefault() const {
	return default_;
}

ParameterEnum::ParameterEnum(const char* name, const char* enumValues, const char* value, const char* prefix) :
		Precursor(name, prefix)
{
	std::string tmp(enumValues);
	std::vector<std::string> v;
	Core::Utils::tokenizeString(v, tmp, ", ");
	for (u32 i = 0; i < v.size(); i++) {
		enum_[v[i]] = i;
	}
	std::string val(value);
	std::map<std::string, u32>::iterator it = enum_.find(val);
	if (it == enum_.end()) {
		std::cerr << "Default value of ParameterEnum must be one of " << enumValues << ", but is " << value << ". Abort." << std::endl;
		exit(1);
	}
	else {
		default_ = it->second;
	}
}

const u32 ParameterEnum::getDefault() const {
	return default_;
}

u32 ParameterEnum::num(const std::string& str) const {
	std::map<std::string, u32>::const_iterator it = enum_.find(str);
	if (it == enum_.end()) {
		std::cerr << "ParameterEnum: " << getName() << "." << getPrefix() << " must be one of ";
		for (std::map<std::string, u32>::const_iterator it=enum_.begin(); it!=enum_.end(); ++it) {
			std::cerr << it->first << ", ";
		}
		std::cerr << "but is " << str << ". Abort." << std::endl;
		exit(1);
		return -1;
	}
	else {
		return it->second;
	}
}

u32 ParameterEnum::num(const char* str) const {
	std::string tmp(str);
	return num(tmp);
}
