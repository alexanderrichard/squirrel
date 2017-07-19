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
 * Log.cc
 *
 *  Created on: 26.03.2014
 *      Author: richard
 */

#include "Log.hh"
#include "Configuration.hh"
#include "Utils.hh"
#include <iostream>
#include <stdlib.h>

using namespace Core;

const ParameterString Log::paramLogFile("log-file", "stdout");

Log* Log::theInstance_ = 0;

u32 Log::indentationLevel() {
	return tags_.size();
}

void Log::indent() {
	for (u32 i = 0; i < indentationLevel(); i++) {
		(*os_) << "  ";
	}
}

void Log::setOutputFile(const char* filename) {
	if (ofs_.is_open()) {
		ofs_.close();
	}
	ofs_.open(filename);
	if (!ofs_.is_open()) {
		std::cerr << "Log: Could not open logfile " << filename << ". Abort." << std::endl;
		exit(1);
	}
	os_ = &ofs_;
}

Log* Log::getInstance() {
	if (theInstance_ == 0) {
		theInstance_ = new Log();
		// check for log file
		std::string filename("stdout");
		if (Configuration::isInitialized()) {
			filename = Configuration::config(Log::paramLogFile);
		}
		if (filename.compare(std::string("stdout")) != 0) {
			Log::getInstance()->setOutputFile(filename.c_str());
		}
		// print header
		(*(Log::getInstance()->os_)) << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>";
	}
	return theInstance_;
}

Log::Log() :
		os_(&(std::cout))
{}

void Log::finalize() {
	// close all open tags
	while (Log::getInstance()->tags_.size() > 0) {
		Log::closeTag();
	}
	// end line
	std::ostream* os;
	os = Log::getInstance()->os_;
	(*os) << std::endl;
	// close log file if any
	if (Log::getInstance()->ofs_.is_open()) {
		Log::getInstance()->ofs_.close();
	}
	// delete the instance
	delete theInstance_;
	theInstance_ = 0;
}

std::ostream& Log::os(const char* msg) {
	std::ostream* os;
	os = Log::getInstance()->os_;
	(*os) << std::endl;
	Log::getInstance()->indent();
	(*os) << msg;
	return (*os);
}

void Log::openTag(const char* tag, const char* description) {
	std::string strTag(tag);
	// replace spaces
	Core::Utils::replaceChar(strTag, ' ', '_');
	// add description to tag
	std::string descr(description);
	std::string msg(strTag);
	if (!descr.empty()) {
		msg.append(" ");
		msg.append(descr);
	}
	Log::os() << "<" << msg << ">";
	Log::getInstance()->tags_.push_back(strTag);
}

void Log::closeTag() {
	std::string tag(Log::getInstance()->tags_.back());
	Log::getInstance()->tags_.pop_back();
	Log::os() << "</" << tag << ">";
}
