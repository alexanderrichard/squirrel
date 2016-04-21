#ifndef MATH_RANDOM_HH_
#define MATH_RANDOM_HH_

#include "Core/CommonHeaders.hh"

namespace Math {

class RandomNumberGenerator
{
private:
	static const Core::ParameterInt paramSeed_;
	static bool isInitialized_;
public:
	static void initializeSRand();
	static void resetSRand();

	/* @return random float between 0 and 1 */
	Float random(bool includingZero = true, bool includingOne = true);
	/* @return random float between a and b */
	Float random(Float a, Float b, bool includingLeftBoundary = true, bool includingRightBoundary = true);
	/* @return random signed integer between a and b */
	s32 randomInt(s32 a, s32 b);
	/* @return random unsigned integer between a and b */
	u32 randomInt(u32 a, u32 b);
};

} // namespace


#endif /* MATH_RANDOM_HH_ */
