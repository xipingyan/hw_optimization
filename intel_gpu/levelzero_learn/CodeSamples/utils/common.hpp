#pragma once

#include <iostream>

#include "log.hpp"
#include "level_zero/ze_api.h"

#define CHECK_RET(RET)                                       \
    if (ZE_RESULT_SUCCESS != RET)                            \
    {                                                        \
        std::cout << "== Fail: return " << RET << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(0);                                             \
    }