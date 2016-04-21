#ifndef CORE_LOG_HH_
#define CORE_LOG_HH_

#include <ostream>
#include <fstream>
#include <string>
#include <vector>
#include "Types.hh"
#include "Parameter.hh"

namespace Core {

/*
 * singleton for logging program information to a file or std::out
 */
class Log
{
private:
	static const ParameterString paramLogFile;

	std::ofstream ofs_; // output file stream
	std::ostream* os_; // output stream for logging
	std::vector<std::string> tags_; // stack containing all currently open tags

	u32 indentationLevel();
	void indent();
	void setOutputFile(const char* filename);

	static Log* theInstance_;
	static Log* getInstance();
	Log();
public:
	/*
	 * @param msg the log message that is written to the output stream
	 * @return a reference to the output stream
	 */
	static std::ostream& os(const char* msg = "");
	/*
	 * opens an XML tag in the log file
	 * @param tag the name of the tag
	 * @param description an optional description of the tag
	 */
	static void openTag(const char* tag, const char* description = "");
	static void openTag(std::string& tag) { Log::openTag(tag.c_str()); }
	/*
	 * close the most recent tag
	 */
	static void closeTag();
	/*
	 * close all open tags, close the output stream, and delete the singleton instance of this class
	 */
	static void finalize();
};

} // namespace


#endif /* CORE_LOG_HH_ */
