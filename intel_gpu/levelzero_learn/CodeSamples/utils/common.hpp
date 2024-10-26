#pragma once

#include "log.hpp"
#include "level_zero/ze_api.h"

#define CHECK_RET(RET)                                       \
    if (ZE_RESULT_SUCCESS != RET)                            \
    {                                                        \
        std::cout << "== Fail: return " << RET << std::endl; \
        exit(0);                                             \
    }
