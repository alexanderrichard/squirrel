#include "Random.hh"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <typeinfo>
#include <math.h>

using namespace Math;

const Core::ParameterInt RandomNumberGenerator::paramSeed_("seed", Types::max<u32>(), "math.random");

bool RandomNumberGenerator::isInitialized_ = false;

void RandomNumberGenerator::initializeSRand() {
	if (!isInitialized_) {
		resetSRand();
	}
	isInitialized_ = true;
}

void RandomNumberGenerator::resetSRand() {
	Core::Log::openTag("math.random");
	u32 seed = Core::Configuration::config(paramSeed_);
	if (seed == Types::max<u32>()) {
		struct timeval t;
		gettimeofday(&t, NULL);
		seed = (time(0) % 1000000) + t.tv_usec;
	}
	Core::Log::os("invoke srand with seed ") << seed;
	Core::Log::closeTag();
	srand(seed);
}

Float RandomNumberGenerator::random(bool includingZero, bool includingOne) {
	if (!isInitialized_)
		initializeSRand();
	Float randVal = (Float)rand() / (Float) RAND_MAX;
	if (typeid(Float) == typeid(f32)) {
		if ((randVal == 0) && (!includingZero))
			randVal = nextafterf(0, 1);
		if ((randVal == 1) && (!includingOne))
			randVal = nextafterf(1, 0);
	}
	else {
		if ((randVal == 0) && (!includingZero))
			randVal = nextafter(0, 1);
		if ((randVal == 1) && (!includingOne))
			randVal = nextafter(1, 0);
	}
	return randVal;
}

Float RandomNumberGenerator::random(Float a, Float b, bool includingLeftBoundary, bool includingRightBoundary) {
	require_le(a, b);
	if (!isInitialized_)
		initializeSRand();
	Float randVal = random() * (b - a) + a;
	if (typeid(Float) == typeid(f32)) {
		if ((randVal == a) && (!includingLeftBoundary))
			randVal = nextafterf(a, b);
		if ((randVal == b) && (!includingRightBoundary))
			randVal = nextafterf(b, a);
	}
	else {
		if ((randVal == a) && (!includingLeftBoundary))
			randVal = nextafter(a, b);
		if ((randVal == b) && (!includingRightBoundary))
			randVal = nextafter(b, a);
	}
	return randVal;
}

s32 RandomNumberGenerator::randomInt(s32 a, s32 b) {
	require_le(a, b);
	if (!isInitialized_)
		initializeSRand();
	return (rand() % (b - a + 1)) + a;
}

u32 RandomNumberGenerator::randomInt(u32 a, u32 b) {
	require_le(a, b);
	if (!isInitialized_)
		initializeSRand();
	return (rand() % (b - a + 1)) + a;
}
