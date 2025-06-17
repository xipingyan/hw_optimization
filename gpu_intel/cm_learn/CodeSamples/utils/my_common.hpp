#pragma once

#include <cstring>
#include <cfloat>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include "my_log.hpp"
// Check all ze function return.
#define CHECK_RET(RET)                                                                                                      \
	if (ZE_RESULT_SUCCESS != RET)                                                                                           \
	{                                                                                                                       \
		std::cout << "== Fail: return " << std::hex << RET << std::dec << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
		exit(0);                                                                                                            \
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

inline bool get_env(std::string env)
{
	auto env_str = std::getenv(env.c_str());
	if (env_str && std::string("1") == env_str)
	{
		return true;
	}
	return false;
}

std::vector<float> init_vec_with_random(const size_t &size)
{
	std::vector<float> data(size);

	// Random number generation setup
	std::random_device rd;									// Seed
	std::mt19937 gen(rd());									// Mersenne Twister engine
	std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Range [0.0, 1.0)

	// Fill the vector with random floats
	for (auto &val : data)
	{
		val = dist(gen);
	}

	// Print the vector
	std::cout << "Random float vector:\n";
	for (const auto &val : data)
	{
		std::cout << val << " ";
	}
	std::cout << std::endl;
	return data;
}

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