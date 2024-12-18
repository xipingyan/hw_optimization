#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <fstream>
#include <vector>

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