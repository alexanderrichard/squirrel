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
