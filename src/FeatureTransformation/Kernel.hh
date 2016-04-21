#ifndef FEATURETRANSFOMATION_KERNEL_HH_
#define FEATURETRANSFOMATION_KERNEL_HH_

#include "Math/Vector.hh"
#include "Features/FeatureReader.hh"
#include "Features/FeatureWriter.hh"

namespace FeatureTransformation {

/*
 * given two feature caches (here called Train and Test), compute a kernel between these two sets of features
 */
class Kernel
{
private:
	static const Core::ParameterEnum paramKernel_;
	enum Kernels { none, linear, histogramIntersection, hellinger, multichannelRbfChiSquare, modifiedChiSquare };
protected:
	Features::FeatureReader featureReaderTrain_;
	Features::FeatureReader featureReaderTest_;
	Features::FeatureWriter featureWriter_;

	bool isInitialized_;
	bool isFinalized_;
public:
	Kernel();
	virtual ~Kernel() {}

	virtual void initialize();
	virtual void finalize();
	/*
	 * apply kernel to the input vector x_i (K(x_i,x_j) for all x_j in train cache) and store result in output
	 */
	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output) = 0;
	/*
	 * apply kernel to all features x_i (K(x_i,x_j) for all x_i in test cache and all x_j in train cache) and write resulting cache
	 */
	virtual void applyKernel();

	static Kernel* createKernel();
};

/*
 * linear kernel
 * K(x,y) = x'y
 */
class LinearKernel : public Kernel
{
public:
	LinearKernel() {}
	virtual ~LinearKernel() {}

	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output);
};

/*
 * HistogramIntersectionKernel
 */
class HistogramIntersectionKernel : public Kernel
{
public:
	HistogramIntersectionKernel() {}
	virtual ~HistogramIntersectionKernel() {}

	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output);
};

/*
 * HellingerKernel
 */
class HellingerKernel : public Kernel
{
public:
	HellingerKernel() {}
	virtual ~HellingerKernel() {}

	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output);
};

/*
 * modified chi-square kernel as used in
 * Efficient Additive Kernels via Explicit Feature Maps, Vedaldi and Zisserman
 */
class ModifiedChiSquareKernel : public Kernel
{
public:
	ModifiedChiSquareKernel() {}
	virtual ~ModifiedChiSquareKernel() {}

	/*
	 * apply kernel to the input vector x_i (K(x_i,x_j) for all x_j in train cache) and store result in output
	 */
	virtual void applyKernel(const Math::Vector<Float>& input, Math::Vector<Float>& output);
};

} // namespace

#endif /* KERNEL_HH_ */
