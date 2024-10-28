#pragma once

#include "log.hpp"
#include "level_zero/ze_api.h"

#define CHECK_RET(RET)                                       \
    if (ZE_RESULT_SUCCESS != RET)                            \
    {                                                        \
        std::cout << "== Fail: return " << RET << std::endl; \
        exit(0);                                             \
    }

inline bool get_device(ze_driver_handle_t& hDriver, ze_device_handle_t& hDevice) {
	// Initialize the driver
	zeInit(0);

	// Discover all the driver instances
	uint32_t driverCount = 0;
	zeDriverGet(&driverCount, nullptr);

	ze_driver_handle_t* allDrivers = (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
	zeDriverGet(&driverCount, allDrivers);

	// Find a driver instance with a GPU device
	// ze_driver_handle_t hDriver = nullptr;
	// ze_device_handle_t hDevice = nullptr;
	for(uint32_t i = 0; i < driverCount; ++i) {
		uint32_t deviceCount = 0;
		zeDeviceGet(allDrivers[i], &deviceCount, nullptr);

		ze_device_handle_t* allDevices = (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
		zeDeviceGet(allDrivers[i], &deviceCount, allDevices);

		for(uint32_t d = 0; d < deviceCount; ++d) {
			ze_device_properties_t device_properties {};
			device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
			zeDeviceGetProperties(allDevices[d], &device_properties);

			if(ZE_DEVICE_TYPE_GPU == device_properties.type) {
				hDriver = allDrivers[i];
				hDevice = allDevices[d];
				break;
			}
		}

		free(allDevices);
		if(nullptr != hDriver) {
			break;
		}
	}

	free(allDrivers);
	if(nullptr == hDevice)
		return false; // no GPU devices found
	return true;
}