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
