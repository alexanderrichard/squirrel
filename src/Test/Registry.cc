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

#include <Test/Registry.hh>
#include <cppunit/extensions/TestFactoryRegistry.h>

using namespace Test;

TestSuiteRegistry* TestSuiteRegistry::instance_ = 0;

TestSuiteRegistry& TestSuiteRegistry::instance()
{
	if (!instance_) {
		instance_ = new TestSuiteRegistry();
		CppUnit::TestFactoryRegistry::getRegistry().registerFactory(instance_);
	}
	return *instance_;
}

/**
 * add a test case to the given test suite.
 */
bool TestSuiteRegistry::addTest(const std::string &module,
		const std::string &suiteName,
		CppUnit::Test *test)
{
	SuiteMap &suites = modules_[module];
	if (suites.find(suiteName) == suites.end()) {
		suites.insert(SuiteMap::value_type(suiteName,
				new CppUnit::TestSuite(suiteName)));
	}
	suites[suiteName]->addTest(test);
	return true;
}

/**
 * generate a CppUnit test case including all registered test cases.
 */
CppUnit::Test* TestSuiteRegistry::makeTest()
{
	CppUnit::TestSuite *allTests = new CppUnit::TestSuite("all");
	for (ModuleMap::const_iterator m = modules_.begin(); m != modules_.end(); ++m) {
		for (SuiteMap::const_iterator i = m->second.begin(); i != m->second.end(); ++i) {
			allTests->addTest(i->second);
		}
	}
	return allTests;
}
