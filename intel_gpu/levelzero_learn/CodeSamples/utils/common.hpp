#pragma once

#include "log.hpp"
#include "level_zero/ze_api.h"

#define CHECK_RET(RET)                                       \
    if (ZE_RESULT_SUCCESS != RET)                            \
    {                                                        \
        std::cout << "== Fail: return " << RET << std::endl; \
        exit(0);                                             \
    }

void list_devices() {
    DEBUG_LOG << "Call zeInit" << std::endl;
    auto r = zeInit(0);
    CHECK_RET(r)

    DEBUG_LOG << "Call zeDriverGet" << std::endl;
    uint32_t count = 0;
    ze_driver_handle_t hDrivers;
    r = zeDriverGet(&count, &hDrivers);
    CHECK_RET(r)

    for (uint32_t i = 0; i <count; i++ ) {
        std::cout << "Driver: " << i << ", " << std::endl;
    }
}