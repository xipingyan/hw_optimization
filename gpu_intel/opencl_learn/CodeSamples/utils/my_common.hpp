#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

#include "my_log.hpp"
#include "level_zero/ze_api.h"

// Check all ze function return.
#define CHECK_RET(RET)                                                                                                      \
    if (ZE_RESULT_SUCCESS != RET)                                                                                           \
    {                                                                                                                       \
        std::cout << "== Fail: return " << std::hex << RET << std::dec << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(0);                                                                                                            \
    }

inline std::string ze_rslt_to_str(ze_result_t r) {
	switch (r)
	{
#define CASE(ITM) case ITM: return #ITM
	CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
	CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
	default:
		return "";
	}
};

#ifndef SUCCESS_OR_TERMINATE
#define SUCCESS_OR_TERMINATE(fun)                                                                                                       \
	{                                                                                                                                   \
		auto ret_fun = fun;                                                                                                             \
		if (ret_fun != ZE_RESULT_SUCCESS)                                                                                               \
		{                                                                                                                               \
			std::cout << "== Fail: return " << std::hex << ret_fun << std::dec << ", " << __FILE__ << ":" << __LINE__ << std::endl;     \
			std::cout << "   " << std::hex << ret_fun << std::dec << " means: " << ze_rslt_to_str(ret_fun) << std::endl; \
			exit(0);                                                                                                                    \
		}                                                                                                                               \
	}
#endif

class CKernelBinFile
{
public:
	// input spv/ocl binary file name.
	CKernelBinFile(std::string fn) {
		std::ifstream file(fn.c_str(), std::ios::binary);
		if (file.is_open()) {
			file.seekg(0, file.end);
			_fileSize = file.tellg();
			file.seekg(0, file.beg);
			_pbuf = (uint8_t*)(malloc(_fileSize));
			if (!_pbuf) {
				std::cout << "== Fail: can't malloc, size: " << _fileSize << std::endl;
				return;
			}
			file.read((char*)_pbuf, _fileSize);
			file.close();
		}
		else {
			std::cout << "== Fail: can't open: " << fn << std::endl;
		}
	}
	CKernelBinFile() = delete;
	using PTR=std::shared_ptr<CKernelBinFile>;
	static PTR createPtr(std::string fn) {
		return std::make_shared<CKernelBinFile>(fn);
	}
	~CKernelBinFile() {
		if (_pbuf) {
			free(_pbuf);
			_pbuf = nullptr;
		}
	}
	size_t _fileSize = 0;
	uint8_t* _pbuf = nullptr;
};

template <typename T>
inline bool is_close(const std::vector<T> &vec1, const std::vector<T> &vec2, const float& thr = 0.0001f)
{
	// 1. Check if the sizes are different
	if (vec1.size() != vec2.size())
	{
		return false;
	}

	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (fabs(vec1[i] - vec2[i]) > thr)
		{
			return false; // Found a differing element
		}
	}

	// If we reach here, sizes are the same and all elements are equal
	return true;
}

template <typename T>
inline void print_diff(const std::vector<T> &vec1, const std::vector<T> &vec2, const float& thr = 0.0001f)
{
	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (fabs(vec1[i] - vec2[i]) > thr)
		{
			std::cout << "   vec1[" << i << "] = " << vec1[i] << ", vec2[" << i << "] = " << vec2[i] << ", diff = " << fabs(vec1[i] - vec2[i]) << std::endl;
		}
	}
}

template <typename T>
inline bool is_same(const std::vector<T> &vec1, const std::vector<T> &vec2) {
	return is_close<T>(vec1, vec2, 0);
}

inline std::vector<float> generate_vec(int sz) {
	// 1. Create a random number generator engine.
    //    `std::mt19937` is a Mersenne Twister engine, which is generally
    //    a good choice for most applications due to its high quality.
    std::random_device rd;
    std::mt19937 gen(rd());

    // 2. Define the distribution.
    //    `std::uniform_real_distribution<float>` ensures numbers are
    //    uniformly distributed between the specified range [0, 1).
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // 3. Create a function object (lambda) to generate a single random number.
    auto rand_float = [&]() { return dis(gen); };

    // 4. Create the vector and fill it.
    //    Here we'll create a vector of 10 random floats.
    std::vector<float> random_floats(sz);
    std::generate(random_floats.begin(), random_floats.end(), rand_float);
	return random_floats;
}

inline size_t tm_diff_ms(std::chrono::time_point<std::chrono::high_resolution_clock> &t1, std::chrono::time_point<std::chrono::high_resolution_clock> &t2)
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
}

inline std::string get_env_str(std::string str_env)
{
	if (std::getenv(str_env.c_str()))
	{
		return std::string(std::getenv(str_env.c_str()));
	}
	return std::string();
}

inline bool get_env_bool(std::string str_env)
{
	if (std::getenv(str_env.c_str()))
	{
		return std::string(std::getenv(str_env.c_str())) == std::string("1") ||
			   std::string(std::getenv(str_env.c_str())) == std::string("true") ||
			   std::string(std::getenv(str_env.c_str())) == std::string("True") ||
			   std::string(std::getenv(str_env.c_str())) == std::string("TRUE");
	}
	return false;
}

inline int get_env_int(std::string str_env)
{
	if (std::getenv(str_env.c_str()))
	{
		return std::atoi(std::getenv(str_env.c_str()));
	}
	return -1;
}