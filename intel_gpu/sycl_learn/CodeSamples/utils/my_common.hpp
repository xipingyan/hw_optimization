#pragma once

#include <cstring>
#include <cfloat>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <cmath>

#include "my_log.hpp"
#include <sycl/sycl.hpp>

// Check all ze function return.
#define CHECK_RET(RET)                                                                                                      \
	if (ZE_RESULT_SUCCESS != RET)                                                                                           \
	{                                                                                                                       \
		std::cout << "== Fail: return " << std::hex << RET << std::dec << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
		exit(0);                                                                                                            \
	}

#define Start_Test() std::cout << "== Start test: " << __FUNCTION__ << std::endl
#define End_Test() std::cout << "== Finish test: " << __FUNCTION__ << std::endl

inline void print_device_beckend(sycl::queue &queue, std::string prefix = " ")
{
	// sycl::queue queue;
	std::cout << prefix << "Using "
			  << queue.get_device().get_info<sycl::info::device::name>()
			  << ", Backend: " << queue.get_backend()
			  << std::endl;
}

template <typename T>
inline bool check_result(std::vector<T> result, std::vector<T> expected, T thr)
{
	bool result_is_expected = true;
	for (size_t i = 0; i < result.size(); i++)
	{
		if (fabs(expected[i] - result[i]) > thr)
		{
			std::cout << "== Result [" << i << "] diff: " << fabs(expected[i] - result[i]) << ", result=" << result[i] << std::endl;
			result_is_expected = false;
		}
	}
	return result_is_expected;
}

template <typename T>
inline bool check_result(T *result, T *expected, size_t len, T thr)
{
	bool result_is_expected = true;
	for (size_t i = 0; i < len; i++)
	{
		if (fabs(expected[i] - result[i]) > thr)
		{
			std::cout << "== Result [" << i << "] diff: " << fabs(expected[i] - result[i]) << ", result=" << result[i] << std::endl;
			result_is_expected = false;
		}
	}
	return result_is_expected;
}

inline size_t tm_diff_ms(std::chrono::time_point<std::chrono::high_resolution_clock> &t1, std::chrono::time_point<std::chrono::high_resolution_clock> &t2)
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}