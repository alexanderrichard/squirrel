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

#include "MatrixToCacheConversion.hh"
#include "Math/Matrix.hh"
#include "Features/FeatureWriter.hh"

using namespace Converter;

const Core::ParameterString MatrixToSingleCacheConverter::paramMatrixFile_("matrix-file", "", "converter.matrix-to-cache-converter");

MatrixToSingleCacheConverter::MatrixToSingleCacheConverter() :
		matrixFile_(Core::Configuration::config(paramMatrixFile_))
{}

void MatrixToSingleCacheConverter::convert() {
	std::cout << "convert..." << std::endl;
	Math::Matrix<Float> matrix;
	matrix.read(matrixFile_, true);
	Features::FeatureWriter featureWriter;
	featureWriter.write(matrix);
	featureWriter.finalize();
}
