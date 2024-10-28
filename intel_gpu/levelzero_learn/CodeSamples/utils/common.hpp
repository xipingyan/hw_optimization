#pragma once

#include "log.hpp"
#include "level_zero/ze_api.h"

#define CHECK_RET(RET)                                       \
    if (ZE_RESULT_SUCCESS != RET)                            \
    {                                                        \
        std::cout << "== Fail: return " << RET << ", " << __FILE__ << ":" << __LINE__ << std::endl; \
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

bool createContext(ze_driver_handle_t hDriver, ze_context_handle_t& hContext) {
	ze_context_desc_t ctxtDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0 };
	auto r = zeContextCreate(hDriver, &ctxtDesc, &hContext);
	CHECK_RET(r)
	return true;
}

bool get_cmd_queue_group_ordinal(ze_device_handle_t hDevice, uint32_t& computeQueueGroupOrdinal) {
	// Discover all command queue groups
	uint32_t cmdqueueGroupCount = 0;
	auto r = zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, nullptr);
	CHECK_RET(r)
	std::cout << "cmdqueueGroupCount = " << cmdqueueGroupCount << std::endl;

	ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)
		malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
	for (uint32_t i = 0; i < cmdqueueGroupCount; i++) {
		cmdqueueGroupProperties[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
		cmdqueueGroupProperties[i].pNext = nullptr;
		r = zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
		CHECK_RET(r)
			std::cout << "  CommandQueueGroup[" << i << "]:" << std::endl;
		std::cout << "    maxMemoryFillPatternSize = " << cmdqueueGroupProperties[i].maxMemoryFillPatternSize << std::endl;
		std::cout << "    numQueues = " << cmdqueueGroupProperties[i].numQueues << std::endl;
	}

	// Find a command queue type that support compute
	computeQueueGroupOrdinal = cmdqueueGroupCount;
	for (uint32_t i = 0; i < cmdqueueGroupCount; ++i) {
		if (cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
			computeQueueGroupOrdinal = i;
			break;
		}
	}
	std::cout << "computeQueueGroupOrdinal = " << computeQueueGroupOrdinal << std::endl;

	free(cmdqueueGroupProperties);

	if (computeQueueGroupOrdinal == cmdqueueGroupCount)
		return false; // no compute queues found
	return true;
}