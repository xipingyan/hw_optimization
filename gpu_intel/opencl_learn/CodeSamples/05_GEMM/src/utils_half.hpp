#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cfloat>
#include <random>
#include <memory>
#include <omp.h>
#include <chrono>

#include "my_common.hpp"

using half = uint16_t;
union FloatBits
{
	float f;
	uint32_t i;
};

inline half floatToHalf(float val)
{
	FloatBits floatBits;
	floatBits.f = val;
	uint32_t f = floatBits.i;

	// Extract components
	uint32_t sign = (f >> 31);
	int32_t exp = (f >> 23) & 0xFF;
	uint32_t mant = (f & 0x7FFFFF);

	// Handle special cases
	if (exp == 0)
	{ // Zero or subnormal
		return (sign << 15);
	}
	if (exp == 0xFF)
	{ // Infinity or NaN
		return (sign << 15) | 0x7C00 | (mant ? 0x0200 : 0);
	}

	// Normal float to half conversion
	exp -= 127;
	if (exp > 15)
	{ // Overflow to infinity
		return (sign << 15) | 0x7C00;
	}
	if (exp < -14)
	{ // Underflow to zero or subnormal
		uint32_t halfMant = mant | 0x800000;
		halfMant >>= (-14 - exp);
		return (sign << 15) | (halfMant >> 13);
	}

	// Normal conversion
	exp += 15;
	mant >>= 13;
	return (sign << 15) | (exp << 10) | mant;
}

// Function to convert half (fp16) to float
inline float halfToFloat(half val)
{
	uint32_t sign = (val >> 15);
	int32_t exp = (val >> 10) & 0x1F;
	uint32_t mant = (val & 0x3FF);

	FloatBits floatBits;

	if (exp == 0)
	{ // Zero or subnormal
		if (mant == 0)
		{
			floatBits.i = (sign << 31);
		}
		else
		{
			// Subnormal
			while (!(mant & 0x400))
			{
				mant <<= 1;
				exp--;
			}
			exp += 1;
			mant &= 0x3FF;
			floatBits.i = (sign << 31) | ((exp + 127) << 23) | (mant << 13);
		}
	}
	else if (exp == 0x1F)
	{ // Infinity or NaN
		floatBits.i = (sign << 31) | 0x7F800000 | (mant << 13);
	}
	else
	{ // Normal half
		floatBits.i = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
	}

	return floatBits.f;
}

inline std::vector<float> vec_to_float(std::vector<half> in)
{
	std::vector<float> out(in.size());
	for (size_t i = 0; i < in.size(); i++)
	{
		out[i] = halfToFloat(in[i]);
	}
	return out;
}