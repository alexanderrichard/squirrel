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

#ifndef CONVERTER_MATRIXTOCACHECONVERSION_HH_
#define CONVERTER_MATRIXTOCACHECONVERSION_HH_

#include "Core/CommonHeaders.hh"

namespace Converter {

/*
 * convert a matrix to a feature cache
 * each row of the matrix is an observation in the feature cache
 * feature dimension is the number of columns
 */
class MatrixToSingleCacheConverter
{
private:
	static const Core::ParameterString paramMatrixFile_;
	std::string matrixFile_;
public:
	MatrixToSingleCacheConverter();
	void convert();
};

} // namespace

#endif /* CONVERTER_MATRIXTOCACHECONVERSION_HH_ */
