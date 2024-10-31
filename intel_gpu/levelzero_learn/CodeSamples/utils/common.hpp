#pragma once

#include <iostream>

#include "log.hpp"
#include "level_zero/ze_api.h"

// Check all ze function return.
#define CHECK_RET(RET)                                                                                                      \
    if (ZE_RESULT_SUCCESS != RET)                                                                                           \
    {                                                                                                                       \
        std::cout << "== Fail: return " << std::hex << RET << std::dec << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(0);                                                                                                            \
    }
