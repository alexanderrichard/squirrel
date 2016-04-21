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

#ifndef CORE_UTILS_HH_
#define CORE_UTILS_HH_

#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <time.h>
#include "Types.hh"

/* require */

#define _require(a, s) { \
if (!(a)) \
{ \
printf("\n\t\033[0;31mREQUIRE\033[0;0m: %s:%d: %s is not true%s. Abort.\n", __FILE__, __LINE__, #a, s); \
exit(1); \
} }

#define _require2(v, a, b) { \
if (!(v)) \
{ \
std::stringstream s; \
s << " (" << a << " vs. " << b << ")"; \
_require(v, s.str().c_str()); \
} }

#define require(a) { _require(a, ""); }

#define require_eq(a,b) { _require2(a == b, a, b); }
#define require_le(a,b) { _require2(a <= b, a, b); }
#define require_ge(a,b) { _require2(a >= b, a, b); }
#define require_lt(a,b) { _require2(a < b, a, b); }
#define require_gt(a,b) { _require2(a > b, a, b); }

/* end require */

namespace Core {

class Utils {
public:
	/*
	 * @param str the string to be tokenized
	 * @param delimiter characters at which the string is to be split
	 * @return vector of strings, each element is a non-empty substring (a token)
	 */
	static void tokenizeString(std::vector<std::string> & result, const std::string& str, const char* delimiter = " ");
	/*
	 * @param str reference to string that is to be manipulated
	 * @param oldCh character to be replaced
	 * @param newCh character that is inserted at the positions of oldCh
	 */
	static void replaceChar(std::string& str, char oldCh, char newCh);
	/*
	 * @param str reference to string that is to be manipulated
	 * @param chars characters to be removed from str
	 */
	static void removeAllOf(std::string& str, const char* chars);
	/*
	 * @return true if the given string ends with ".bin"
	 */
	static bool isBinary(const std::string& filename);
	/*
	 * @return true if the given string ends with ".gz"
	 */
	static bool isGz(const std::string& filename);
	/*
	 * @param str string to append the suffix to
	 * @param suffix append suffix to this string
	 */
	static void appendSuffix(std::string& str, const std::string& suffix);

	/*
	 * computes difference between start and end time
	 */
	static f64 timeDiff(timeval& start, timeval& end);
};

} // namespace

#endif /* CORE_UTILS_HH_ */
