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

#include "Utils.hh"
#include "Types.hh"
#include <sys/time.h>

using namespace Core;

void Utils::print_stacktrace() {
	u32 n_pointers;
	u32 buffer_size = 100;
	void *buffer[buffer_size];
	char **strings;

	n_pointers = backtrace(buffer, buffer_size);
	strings = backtrace_symbols(buffer, n_pointers);
	if (strings == NULL) {
		std::cerr << "backtrace: no backtrace symbols." << std::endl;
		exit(1);
	}

	for (u32 i = 0; i < n_pointers; i++) {
		std::cout << strings[i] << std::endl;
	}
	free(strings);
}

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

void Utils::appendSuffix(std::string& filename, const std::string& suffix) {
	if (Core::Utils::isGz(filename)) {
		filename = filename.substr(0, filename.size() - 3).append(suffix).append(".gz");
	}
	else if (Core::Utils::isBinary(filename)) {
		filename = filename.substr(0, filename.size() - 4).append(suffix).append(".bin");
	}
	else {
		filename = filename.append(suffix);
	}
}

f64 Utils::timeDiff(timeval& start, timeval& end) {
	f64 diff = 0;
	diff = end.tv_sec - start.tv_sec;
	diff += (end.tv_usec - start.tv_usec) / 1000000.0;
	return diff;
}

/*
 * some useful converters
 */
void Utils::copyCVMatToMemory(const cv::Mat& image, Float* dest) {
#ifdef MODULE_OPENCV

	for(u32 r = 0; r < (u32)image.rows; r++) {
		// gray scale
		if (image.channels() == 1) {
			const cv::Vec<float, 1>* row = image.ptr< cv::Vec<float, 1> >(r);
			for(u32 c = 0; c < (u32)image.cols; c++) {
				for(u32 ch = 0; ch < (u32)image.channels(); ch++) {
					dest[ch*image.rows*image.cols + r*image.cols + c] = row[c].val[ch];
				}
			}
		}
		// rgb
		else if (image.channels() == 3) {
			const cv::Vec<float, 3>* row = image.ptr< cv::Vec<float, 3> >(r);
			for(u32 c = 0; c < (u32)image.cols; c++) {
				for(u32 ch = 0; ch < (u32)image.channels(); ch++) {
					dest[ch*image.rows*image.cols + r*image.cols + c] = row[c].val[ch];
				}
			}
		}
	}
#else
	Core::Error::msg("Utils::copyCVMatToMemory requires OpenCV but binary is not compiled with OpenCV support.") << Core::Error::abort;
#endif
}

void Utils::copyMemoryToCVMat(const Float* src, cv::Mat& image) {
#ifdef MODULE_OPENCV
	std::vector<cv::Mat> img_ch;
	for (u32 ch = 0; ch < (u32)image.channels(); ch++) {
		img_ch.push_back(cv::Mat(image.rows, image.cols, CV_32FC1,
				const_cast<Float*>(src) + ch * image.rows * image.cols));
	}
	cv::merge(img_ch, image);
#else
	Core::Error::msg("Utils::copyMemoryToCVMat requires OpenCV but binary is not compiled with OpenCV support.") << Core::Error::abort;
#endif
}

/*
 * Timer
 */
Utils::Timer::Timer() :
		startVal_(0),
		elapsedTime_(0)
{}

void Utils::Timer::run() {
	gettimeofday(&t, NULL);
	startVal_ = (t.tv_sec * 1000) + (t.tv_usec / 1000);
}

void Utils::Timer::stop() {
	gettimeofday(&t, NULL);
	elapsedTime_ += ((t.tv_sec * 1000) + (t.tv_usec / 1000)) - startVal_;
}

void Utils::Timer::reset() {
	elapsedTime_ = 0;
}

Float Utils::Timer::time() {
	return (Float)elapsedTime_ / 1000.0;
}
