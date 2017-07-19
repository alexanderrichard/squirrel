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
 * Configuration.hh
 *
 *  Created on: 24.03.2014
 *      Author: richard
 */

#ifndef CORE_CONFIGURATION_HH_
#define CORE_CONFIGURATION_HH_

#include <string>
#include <map>
#include <vector>
#include "Types.hh"
#include "Parameter.hh"

/*
 * Configuration
 *
 * ***
 * *** (1) command line parameters
 * ***
 *
 * command line parameters can be given in the format
 *     --prefix-path.parameter-name=value
 * or
 *     --*.parameter-name=value
 * whitespaces and tabs are not allowed for command line parameters
 *
 *
 * ***
 * *** (2) config files
 * ***
 *
 * config files are read line-by-line with the following specification:
 *
 * (a) comments
 * "#" starts a comment
 *     # this is a comment
 *     parameter-name = parameter-value # this is another comment
 * comments can be anywhere in the config
 * everything after a "#" is ignored
 *
 * (b) includes
 * other config files can be included via
 *     include <filename>
 *
 * (c) parameters
 * parameters can be specified via
 *     prefix-path.parameter-name = parameter-value
 * or via
 *     *.parameter-name = parameter-value
 * first, it is searched for <prefix-path.parameter-name>, then for <*.parameter-name>
 * if none of them are found, the default value is used for the parameter
 *
 * (d) global prefixes
 * global prefixes can be defined via
 *     [global-prefix]
 * all following parameter specifications are assumed to have this global prefix, e.g.
 *     [global-prefix]
 *     parameter-name = parameter-value
 * is interpreted as
 *     global-prefix.parameter-path.parameter-name = parameter-value
 * the global prefix is valid until a new one is defined
 * "[]" resets the global prefix, i.e.
 *     []
 *     parameter-path.parameter-name = parameter-value
 * is interpreted as
 *     parameter-path.parameter-name = parameter-value
 *
 * (e) whitespaces and tabs
 * whitespaces and tabs are ignored
 *
 */
namespace Core {

class Configuration
{
private:
	/* ConfigTree */
	class ConfigTree {
	public:
		struct Node {
			std::string key_;
			std::string value_;
			std::vector<Node> successors_;
			// hasValue_ is needed to determine if search has to be continued in wildcard branch
			// (value might be "" and, thus, is not sufficient to determine this)
			bool hasValue_;
		};
	private:
		Node root_;
		bool isLeaf(Node& node);
		void findWildcardValue(std::string& key, std::string& result);
		void parsePath(const std::vector<std::string>& path);
		void decodePath(std::vector<std::string>& path, Node& node, std::string& result);
		void insertParameter(std::vector<std::string>& path, std::string& value, Node& node);
	public:
		ConfigTree();
		Node& getRoot() { return root_; }
		std::string decodePath(std::vector<std::string>& path);
		void insertParameter(std::vector<std::string>& path, std::string& value);
	};
	/* end ConfigTree */

private:
	// keep track of original input parameters to restore initial config with softReset()
	static int _argc;
	static const char** _argv;

	ConfigTree configTree_;
	std::string prefix_;	// prefix to add to all parameter specifications

	std::string getParameterValue(const std::string& key, const char* prefix = "");
	/*
	 * adds command line parameters to configTree
	 * parameters must be of the form binary-name --<param-path.name=value>, e.g. --*.n=10 or --path.x=5
	 */
	void addCommandLineParameter(const int argc, const char* argv[]);
	void readConfigFile(const char* filename);

	void printNode(ConfigTree::Node& node, u32 indentation);

	static Configuration* theInstance_;
	static bool isInitialized_;
	Configuration() {};
	~Configuration() {}
	static Configuration* getInstance();
public:
	static void initialize();
	static void initialize(const int argc, const char* argv[], bool printLogTree = true);

	static bool isInitialized();

	/* reset is a useful option for some tests */
	static void reset();

	/* config returns value of associated parameter */
	static s32 config(const ParameterInt& param, const char* prefix);
	static f32 config(const ParameterFloat& param, const char* prefix);
	static char config(const ParameterChar& param, const char* prefix);
	static bool config(const ParameterBool& param, const char* prefix);
	static std::string config(const ParameterString& param, const char* prefix);
	static std::vector<s32> config(const ParameterIntList& param, const char* prefix);
	static std::vector<f32> config(const ParameterFloatList& param, const char* prefix);
	static std::vector<char> config(const ParameterCharList& param, const char* prefix);
	static std::vector<bool> config(const ParameterBoolList& param, const char* prefix);
	static std::vector<std::string> config(const ParameterStringList& param, const char* prefix);
	static u32 config(const ParameterEnum& param, const char* prefix);

	/* this variant uses const std::string instead of const char* */
	static s32 config(const ParameterInt& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static f32 config(const ParameterFloat& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static char config(const ParameterChar& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static bool config(const ParameterBool& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::string config(const ParameterString& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::vector<s32> config(const ParameterIntList& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::vector<f32> config(const ParameterFloatList& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::vector<char> config(const ParameterCharList& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::vector<bool> config(const ParameterBoolList& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static std::vector<std::string> config(const ParameterStringList& param, const std::string& prefix) { return config(param, prefix.c_str()); }
	static u32 config(const ParameterEnum& param, const std::string& prefix) { return config(param, prefix.c_str()); }

	/* this variant uses the prefix defined by the parameter */
	static s32 config(const ParameterInt& param) { return config(param, param.getPrefix().c_str()); }
	static f32 config(const ParameterFloat& param) { return config(param, param.getPrefix().c_str()); }
	static char config(const ParameterChar& param) { return config(param, param.getPrefix().c_str()); }
	static bool config(const ParameterBool& param) { return config(param, param.getPrefix().c_str()); }
	static std::string config(const ParameterString& param) { return config(param, param.getPrefix().c_str()); }
	static std::vector<s32> config(const ParameterIntList& param) { return config(param, param.getPrefix().c_str()); }
	static std::vector<f32> config(const ParameterFloatList& param) { return config(param, param.getPrefix().c_str()); }
	static std::vector<char> config(const ParameterCharList& param) { return config(param, param.getPrefix().c_str()); }
	static std::vector<bool> config(const ParameterBoolList& param) { return config(param, param.getPrefix().c_str()); }
	static std::vector<std::string> config(const ParameterStringList& param) { return config(param, param.getPrefix().c_str()); }
	static u32 config(const ParameterEnum& param) { return config(param, param.getPrefix().c_str()); }

	/* manipulate parameters in the config tree */
	/* should only be used for test cases */
	static void setParameter(const char* path, const char* value);

	static void printConfigTreeToLogChannel();
};

} // namespace

#endif /* CORE_CONFIGURATION_HH_ */
