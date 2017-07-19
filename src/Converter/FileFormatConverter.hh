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
 * FileFormatConverter.hh
 *
 *  Created on: May 5, 2017
 *      Author: richard
 */

#ifndef CONVERTER_FILEFORMATCONVERTER_HH_
#define CONVERTER_FILEFORMATCONVERTER_HH_

#include "Core/CommonHeaders.hh"

namespace Converter {

class FileFormatConverter
{
private:
	static const Core::ParameterEnum paramFileType_;
	static const Core::ParameterString paramInputFile_;
	static const Core::ParameterString paramOutputFile_;
	enum FileType { vector, matrix, cache };
	std::string input_;
	std::string output_;
	FileType type_;
	void convertVector();
	void convertMatrix();
	void convertCache();
public:
	FileFormatConverter();
	virtual ~FileFormatConverter() {}
	virtual void convert();
};

} // namespace

#endif /* CONVERTER_FILEFORMATCONVERTER_HH_ */
