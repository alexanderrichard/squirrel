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
 * LengthModel.hh
 *
 *  Created on: May 17, 2017
 *      Author: richard
 */

#ifndef HMM_LENGTHMODEL_HH_
#define HMM_LENGTHMODEL_HH_

#include <Core/CommonHeaders.hh>
#include <Math/Vector.hh>
#include <Math/Matrix.hh>

namespace Hmm {

class LengthModel
{
private:
	static const Core::ParameterEnum paramLengthModelType_;
	static const Core::ParameterBool paramIsFramewise_;
public:
	enum LengthModelType { none, poisson, threshold, linearDecay, monotoneGaussian };
protected:
	LengthModelType type_;
	bool isFramewise_;
public:
	LengthModel();
	virtual ~LengthModel() {}
	LengthModelType type() const { return type_; }
	bool isFramewise() const { return isFramewise_; }
	virtual void initialize() {}
	virtual Float frameScore(u32 length, u32 state) { return 0.0; }
	virtual Float segmentScore(u32 length, u32 state) { return 0.0; }

	/*
	 * factory
	 */
	static LengthModel* create();
};

/*
 * Poisson length model
 * if used as framewise model, 0 penalty until \lambda, then decaying like a Poisson distribution
 */
class PoissonLengthModel : public LengthModel
{
private:
	static const Core::ParameterString paramLengthModelFile_;
	static const Core::ParameterBool paramRescale_;
	typedef LengthModel Precursor;
protected:
	std::string lengthModelFile_;
	bool rescale_;
	Math::Vector<Float> lambda_;
	Math::Vector<Float> rescalingFactor_;
	bool isInitialized_;
public:
	PoissonLengthModel();
	virtual ~PoissonLengthModel() {}
	virtual void initialize();
	virtual Float frameScore(u32 length, u32 state);
	virtual Float segmentScore(u32 length, u32 state);
};

/*
 * threshold length model
 * no penalty until threshold, -inf if length > threshold
 */
class ThresholdLengthModel : public LengthModel
{
private:
	typedef LengthModel Precursor;
	static const Core::ParameterFloat paramThreshold_;
	static const Core::ParameterFloat paramEpsilon_;
protected:
	Float threshold_;
	Float epsilon_;
public:
	ThresholdLengthModel();
	virtual ~ThresholdLengthModel() {}
	virtual Float frameScore(u32 length, u32 state);
	virtual Float segmentScore(u32 length, u32 state);
};

/*
 * linear decay length model
 * probability 1.0 until threshold, linear decay from 1.0 to 0.0 between threshold and 2 * threshold
 */
class LinearDecayLengthModel : public ThresholdLengthModel
{
private:
	typedef ThresholdLengthModel Precursor;
public:
	LinearDecayLengthModel();
	virtual ~LinearDecayLengthModel() {}
	virtual Float frameScore(u32 length, u32 state);
	virtual Float segmentScore(u32 length, u32 state);
};

/*
 * monotone Gaussian length model
 * Gaussian model with mean=0 and stddev=avg-length
 */
class MonotoneGaussianLengthModel : public LengthModel
{
private:
	typedef LengthModel Precursor;
	static const Core::ParameterFloat paramMeanLength_;
	Float meanLength_;
public:
	MonotoneGaussianLengthModel();
	virtual ~MonotoneGaussianLengthModel() {}
	virtual Float frameScore(u32 length, u32 state);
	virtual Float segmentScore(u32 length, u32 state);
};

} // namespace

#endif /* HMM_LENGTHMODEL_HH_ */
