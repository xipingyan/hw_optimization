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
	uint32_t driverCount = 0;
	r = zeDriverGet(&driverCount, nullptr);
	CHECK_RET(r)
	DEBUG_LOG << "Found GPU driverCount = " << driverCount << std::endl;

	ze_driver_handle_t* allDrivers = (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
	zeDriverGet(&driverCount, allDrivers);

	ze_driver_handle_t hDriver = nullptr;
	ze_device_handle_t hDevice = nullptr;
	for (uint32_t i = 0; i < driverCount; i++) {
		uint32_t deviceCount = 0;
    	zeDeviceGet(allDrivers[i], &deviceCount, nullptr);
		std::cout << "  Driver: " << i << ", deviceCount = " << deviceCount << std::endl;

		ze_device_handle_t* allDevices = (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
    	zeDeviceGet(allDrivers[i], &deviceCount, allDevices);

		for(uint32_t d = 0; d < deviceCount; ++d) {
			ze_device_properties_t device_properties {};
			device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
			zeDeviceGetProperties(allDevices[d], &device_properties);
			if(ZE_DEVICE_TYPE_GPU == device_properties.type) {
				hDriver = allDrivers[i];
				hDevice = allDevices[d];
				std::cout << "    Device: " << d << ", device name = " << device_properties.name << std::endl;
			}
		}

		free(allDevices);
		if(nullptr != hDriver) {
			break;
		}
	}

	free(allDrivers);
}