#ifndef FEATURETRANSFORMATION_PRINCIPALCOMPONENTANALYSIS_HH_
#define FEATURETRANSFORMATION_PRINCIPALCOMPONENTANALYSIS_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>
#include <Features/FeatureReader.hh>

namespace FeatureTransformation {

/*
 * compute a PCA based on the given input feature cache
 * both, eigen-decomposition and decomposition via svd are available
 */
class PrincipalComponentAnalysis
{
private:
	static const Core::ParameterString paramMeanFile_;
	static const Core::ParameterString paramPcaMatrixFile_;
	static const Core::ParameterInt paramNumberOfPrincipalComponents_;
	static const Core::ParameterBool paramWhitening_;
	static const Core::ParameterBool paramEstimateMean_;
	static const Core::ParameterEnum paramDecompositionMethod_;
	enum DecompositionMethod { eigen, svd };

	DecompositionMethod decompositionMethod_;
	std::string meanFile_;
	std::string pcaMatrixFile_;
	u32 nPrincipalComponents_;
	bool whitening_;
	bool estimateMean_;

	Math::Vector<Float> mean_;
	Math::Matrix<Float> dataMatrix_;
	Math::Vector<Float> eigenvalues_;
	Math::Matrix<Float> eigenvectors_;

	Features::FeatureReader featureReader_;
private:
	void estimateMean();
	void estimateScatterMatrix();
	void computeDataMatrix();
	void eigenDecomposition();
	void singularValueDecomposition();
	void estimatePca();
public:
	PrincipalComponentAnalysis();
	~PrincipalComponentAnalysis() {}
	void initialize();
	void estimate();
};

} // namespace

#endif /* FEATURETRANSFORMATION_PRINCIPALCOMPONENTANALYSIS_HH_ */
