#include "Utils.hh"
#include "Types.hh"

using namespace Core;

void Utils::tokenizeString(std::vector<std::string>& result, const std::string& str, const char* delimiter)
{
    size_t start = 0, end = 0;
    result.clear();
    std::string delim = std::string(delimiter);

    while (end != std::string::npos)
    {
        end = str.find_first_of(delim, start);
        if (str.substr(start, (end == std::string::npos) ? std::string::npos : end - start).size() > 0) {
        	result.push_back(str.substr(start, (end == std::string::npos) ? std::string::npos : end - start));
        }
        start = end + 1;
    }
}

void Utils::replaceChar(std::string& str, char oldCh, char newCh) {
	for (u32 i = 0; i < str.size(); i++) {
		if (str[i] == oldCh) {
			str[i] = newCh;
		}
	}
}

void Utils::removeAllOf(std::string& str, const char* chars) {
	std::size_t pos = str.find_first_of(chars);
	while (pos != std::string::npos) {
		str.erase(str.begin() + pos);
		pos = str.find_first_of(chars);
	}
}

bool Utils::isBinary(const std::string& filename) {
	std::size_t pos = filename.find_last_of(".");
	std::string str = "";
	if (pos < filename.npos)
		str = filename.substr(pos);
	if (str.compare(".bin") == 0)
		return true;
	else
		return false;
}

bool Utils::isGz(const std::string& filename) {
	std::size_t pos = filename.find_last_of(".");
	std::string str = "";
	if (pos < filename.npos)
		str = filename.substr(pos);
	if (str.compare(".gz") == 0)
		return true;
	else
		return false;
}

void Utils::appendSuffix(std::string& str, const std::string& suffix) {
	if (Core::Utils::isGz(str)) {
		str = str.substr(0, str.size() - 3).append(suffix).append(".gz");
	}
	else if (Core::Utils::isBinary(str)) {
		str = str.substr(0, str.size() - 4).append(suffix).append(".bin");
	}
	else {
		str = str.append(suffix);
	}
}

f64 Utils::timeDiff(timeval& start, timeval& end) {
	f64 diff = 0;
	diff = end.tv_sec - start.tv_sec;
	diff += (end.tv_usec - start.tv_usec) / 1000000.0;
	return diff;
}
