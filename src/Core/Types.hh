#ifndef CORE_TYPES_HH_
#define CORE_TYPES_HH_

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned int u32;
typedef signed int s32;
typedef unsigned long int u64;
typedef signed long int s64;
typedef float f32;
typedef double f64;

// use this type to switch between f32 and f64
typedef f32 Float;

class Types
{
public:
	template<typename T>
	static const T min();

	template<typename T>
	static const T max();

	template<typename T>
	static const T absMin();

	template<typename T>
	static const bool isNan(T val);

	template<typename T>
	static const T inf();
};

#endif /* CORE_TYPES_HH_ */
