#include "Configuration.hh"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include "Utils.hh"
#include "Log.hh"

using namespace Core;

Configuration::ConfigTree::ConfigTree() {
	root_.key_ = "";
	root_.value_ = "";
}

bool Configuration::ConfigTree::isLeaf(Node& node) {
	return node.successors_.empty();
}

void Configuration::ConfigTree::findWildcardValue(std::string& key, std::string& result) {
	// search for '*'
	for (u32 succ = 0; succ < root_.successors_.size(); succ++) {
		if (root_.successors_[succ].key_.compare(std::string("*")) == 0) {
			// search for '*.key'
			for (u32 succsucc = 0; succsucc < root_.successors_[succ].successors_.size(); succsucc++) {
				if (root_.successors_[succ].successors_[succsucc].key_.compare(key) == 0) {
					require(isLeaf(root_.successors_[succ].successors_[succsucc]));
					result = std::string(root_.successors_[succ].successors_[succsucc].value_);
					return;
				}
			}
		}
	}
}

void Configuration::ConfigTree::parsePath(const std::vector<std::string>& path) {
	for (u32 i = 0; i < path.size(); i++) {
		if (path[i].find_first_of(" #=") != std::string::npos) {
			std::cerr << "configuration: path contains invalid characters: " << path[i] << ". Abort." << std::endl;
			exit(1);
		}
	}
}

void Configuration::ConfigTree::decodePath(std::vector<std::string>& path, Node& node, std::string& result) {
	if (path.size() == 0) return;

	// find first element of path in tree induced by node
	for (u32 succ = 0; succ < node.successors_.size(); succ++) {
		if (node.successors_[succ].key_.compare(path[0]) == 0) {
			if ((path.size() == 1) && node.successors_[succ].hasValue_) {
				result = std::string(node.successors_[succ].value_);
			}
			else {
				path.erase(path.begin());
				decodePath(path, node.successors_[succ], result);
			}
			return;
		}
	}
}

std::string Configuration::ConfigTree::decodePath(std::vector<std::string>& path) {
	parsePath(path);
	std::string result = "";
	std::string key = std::string(path.back());
	decodePath(path, root_, result);
	// if no exact path found, look in wildcard branch
	if (result.compare(std::string("")) == 0) {
		findWildcardValue(key, result);
	}
	return result;
}

void Configuration::ConfigTree::insertParameter(std::vector<std::string>& path, std::string& value, Node& node) {
	if (path.size() == 0) {
		node.value_ = std::string(value);
		node.hasValue_ = true;
		return;
	}

	// find first element of path in tree induced by node
	for (u32 succ = 0; succ < node.successors_.size(); succ++) {
		if (node.successors_[succ].key_.compare(path[0]) == 0) {
			path.erase(path.begin());
			insertParameter(path, value, node.successors_[succ]);
			return;
		}
	}
	// if first element of path not available, insert it
	Node n;
	n.key_ = path[0];
	n.value_ = "";
	n.hasValue_ = false;
	node.successors_.push_back(n);
	path.erase(path.begin());
	insertParameter(path, value, node.successors_.back());
}

void Configuration::ConfigTree::insertParameter(std::vector<std::string>& path, std::string& value) {
	parsePath(path);
	insertParameter(path, value, root_);
}

std::string Configuration::getParameterValue(const std::string& key, const char* prefix) {
	std::vector<std::string> path;
	std::string tmp = std::string(prefix);
	Core::Utils::tokenizeString(path, tmp, ".");
	path.push_back(key);
	return configTree_.decodePath(path);
}

void Configuration::addCommandLineParameter(const int argc, const char* argv[]) {
	// extract and add all parameter except for binary name
	for (s32 i = 1; i < argc; i++) {
		std::string tmp(argv[i]);
		tmp.erase(0, tmp.find_first_not_of("-")); // erase --
		std::vector<std::string> argument;
		Core::Utils::tokenizeString(argument, tmp, "=");
		if (argument.size() != 2) {
			std::cerr << tmp << " is an invalid parameter definition. Abort." << std::endl;
			exit(1);
		}
		std::vector<std::string> path;
		Core::Utils::tokenizeString(path, argument[0], ".");
		configTree_.insertParameter(path, argument[1]);
	}
}

void Configuration::readConfigFile(const char* filename) {
	std::ifstream configFile(filename);
	if (!configFile.is_open()) {
		std::cerr << "Failed to open config file " << filename << ". Abort." << std::endl;
		exit(1);
	}

	// reset prefix when starting to read a config file
	prefix_.clear();

	std::string line;
	while (std::getline(configFile, line)) {
		// remove comments
		if (line.find_first_of("#") != std::string::npos) {
			line.erase(line.find_first_of("#"), std::string::npos);
		}
		// remove whitespaces and tabs
		Core::Utils::removeAllOf(line, " \t");
		// tokenize
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, line, "=");
		// skip empty lines
		if (tmp.size() > 0) {
			// process line if one ([prefix] or include<file> (whitespace is removed!)) or two tokens left (path and value)
			if (tmp.size() > 2) {
				std::cerr << line << " is an invalid parameter definition. Abort." << std::endl;
				exit(1);
			}
			/* if only one token ([prefix] or include<file> (whitespace is removed!)) */
			if (tmp.size() == 1) {
				// if "include" read included file
				if (tmp[0].substr(0,7).compare(std::string("include")) == 0) {
					readConfigFile(tmp[0].substr(7).c_str());
					// reset prefix
					prefix_.clear();
				}
				// check for correct format for [...]
				else if ((tmp[0].at(0) == '[') && (tmp[0].at(tmp[0].size() - 1) == ']')) {
					prefix_ = tmp[0].substr(1, tmp[0].size() - 2);
				}
				else {
					std::cerr << line << " is an invalid parameter definition. Abort." << std::endl;
					exit(1);
				}
			}
			/* if two tokens */
			if (tmp.size() == 2) {
				// tokenize path
				std::vector<std::string> path;
				std::string pathAsString = (prefix_.empty() ? "" : prefix_ + ".") + tmp[0];
				Core::Utils::tokenizeString(path, pathAsString, ".");
				configTree_.insertParameter(path, tmp[1]);
			}
		}
	}

	configFile.close();
}

Configuration* Configuration::theInstance_ = 0;
bool Configuration::isInitialized_ = false;

Configuration* Configuration::getInstance() {
	// create singleton if necessary
	if (Configuration::theInstance_ == 0) {
		Configuration::theInstance_ = new Configuration();
	}
	return theInstance_;
}

void Configuration::initialize() {
	const char** dummy = 0;
	initialize(0, dummy);
}

void Configuration::initialize(const int argc, const char* argv[]) {
	// search for the parameter specifying the config file
	for (u32 i = 0; i < (u32)argc; i++) {
		std::string tmp(argv[i]);
		if (tmp.substr(0,9).compare("--config=") == 0) {
			getInstance()->readConfigFile(tmp.substr(9).c_str());
			break;
		}
		else if (tmp.substr(0,11).compare("--*.config=") == 0) {
			getInstance()->readConfigFile(tmp.substr(11).c_str());
			break;
		}
	}
	// add the command line parameters
	// note: command line parameters may overwrite values specified in config file
	Configuration::getInstance()->addCommandLineParameter(argc, argv);
	Configuration::isInitialized_ = true;
	// print tree to log channel
	printConfigTreeToLogChannel();
}

void Configuration::reset() {
	if (theInstance_) {
		delete theInstance_;
		theInstance_ = 0;
		isInitialized_ = false;
	}
}

bool Configuration::isInitialized() {
	return isInitialized_;
}

s32 Configuration::config(const ParameterInt& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		return atoi(value.c_str());
	}
}

f32 Configuration::config(const ParameterFloat& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		return atof(value.c_str());
	}
}

char Configuration::config(const ParameterChar& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		return value.c_str()[0];
	}
}

bool Configuration::config(const ParameterBool& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		if ((value.compare(std::string("true")) == 0) ||
			(value.compare(std::string("TRUE")) == 0) ||
			(value.compare(std::string("yes")) == 0) ||
			(value.compare(std::string("1")) == 0))	{
			return true;
		}
		else {
			return false;
		}
	}
}

std::string Configuration::config(const ParameterString& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		return value;
	}
}

std::vector<s32> Configuration::config(const ParameterIntList& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, value, ",");
		std::vector<s32> list;
		for (u32 i = 0; i < tmp.size(); i++) {
			list.push_back(atoi(tmp[i].c_str()));
		}
		return list;
	}
}

std::vector<f32> Configuration::config(const ParameterFloatList& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, value, ",");
		std::vector<f32> list;
		for (u32 i = 0; i < tmp.size(); i++) {
			list.push_back(atof(tmp[i].c_str()));
		}
		return list;
	}
}

std::vector<char> Configuration::config(const ParameterCharList& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, value, ",");
		std::vector<char> list;
		for (u32 i = 0; i < tmp.size(); i++) {
			list.push_back(tmp[i].c_str()[0]);
		}
		return list;
	}
}

std::vector<bool> Configuration::config(const ParameterBoolList& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		std::vector<std::string> tmp;
		Core::Utils::tokenizeString(tmp, value, ",");
		std::vector<bool> list;
		for (u32 i = 0; i < tmp.size(); i++) {
			if ((tmp[i].compare(std::string("true")) == 0) ||
				(tmp[i].compare(std::string("TRUE")) == 0) ||
				(tmp[i].compare(std::string("yes")) == 0) ||
				(tmp[i].compare(std::string("1")) == 0)) {
				list.push_back(true);
			}
			else {
				list.push_back(false);
			}
		}
		return list;
	}
}

std::vector<std::string> Configuration::config(const ParameterStringList& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		std::vector<std::string> list;
		Core::Utils::tokenizeString(list, value, ", ");
		return list;
	}
}

u32 Configuration::config(const ParameterEnum& param, const char* prefix) {
	require(Configuration::isInitialized_);
	std::string value = Configuration::getInstance()->getParameterValue(param.getName(), prefix);
	if (value.empty()) {
		return param.getDefault();
	}
	else {
		return param.num(value);
	}
}

void Configuration::printNode(ConfigTree::Node& node, u32 indentation) {
	std::stringstream s;
	for (u32 i = 0; i < indentation; i++)
		s << "  ";
	s << " |-" << node.key_;
	if (node.hasValue_)
		s << ":" << node.value_;
	Log::os() << s.str();
	for (u32 succ = 0; succ < node.successors_.size(); succ++) {
		printNode(node.successors_[succ], indentation + 1);
	}
}

void Configuration::setParameter(const char* path, const char* value) {
	// a manually set parameter is a manual initialization
	Configuration::isInitialized_ = true;
	std::string tmp(path);
	std::vector<std::string> vec;
	Core::Utils::tokenizeString(vec, tmp, ".");
	std::string val(value);
	Configuration::getInstance()->configTree_.insertParameter(vec, val);
}

void Configuration::printConfigTreeToLogChannel() {
	Log::openTag("configuration-tree");
	Configuration::getInstance()->printNode(Configuration::getInstance()->configTree_.getRoot(), 0);
	Log::closeTag();
}
