#include <iostream>
#include <level_zero/ze_api.h>

#include "my_common.hpp"

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
		DEBUG_LOG << "  Driver: " << i << ", deviceCount = " << deviceCount << std::endl;

		ze_device_handle_t* allDevices = (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
    	zeDeviceGet(allDrivers[i], &deviceCount, allDevices);

		for(uint32_t d = 0; d < deviceCount; ++d) {
			ze_device_properties_t device_properties {};
			device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
			zeDeviceGetProperties(allDevices[d], &device_properties);
			if(ZE_DEVICE_TYPE_GPU == device_properties.type) {
				hDriver = allDrivers[i];
				hDevice = allDevices[d];
				DEBUG_LOG << "    Device: " << d << ", device name = " << device_properties.name << std::endl;
#define PRINT_PROP(PROPERITY) DEBUG_LOG << "      " << #PROPERITY << " = " << device_properties.PROPERITY << std::endl
#define PRINT_PROP_MEM(PROPERITY) DEBUG_LOG << "      " << #PROPERITY << " = " << device_properties.PROPERITY /1024/1024/1024. << " GB" << std::endl
				PRINT_PROP(type);
				PRINT_PROP(vendorId);
				PRINT_PROP(deviceId);

				PRINT_PROP(flags);
				PRINT_PROP(subdeviceId);

				PRINT_PROP(coreClockRate);
				PRINT_PROP_MEM(maxMemAllocSize);
				PRINT_PROP(maxHardwareContexts);
				PRINT_PROP(maxCommandQueuePriority);
				PRINT_PROP(subdeviceId);

				PRINT_PROP(numThreadsPerEU);
				PRINT_PROP(physicalEUSimdWidth);
				PRINT_PROP(numEUsPerSubslice);
				PRINT_PROP(numSubslicesPerSlice);
				PRINT_PROP(numSlices);
				PRINT_PROP(timerResolution);
				PRINT_PROP(subdeviceId);
			}
		}

		free(allDevices);
		if(nullptr != hDriver) {
			break;
		}
	}

	free(allDrivers);
}

/*
Before you use your GPU to do work, you should know the 
most essential things about its capabilities.
*/
int main()
{
	std::cout << "Hello LevelZero, start to list all drivers and devices." << std::endl;
	list_devices();
	std::cout << "Done." << std::endl;
	return 0;
}

/*
Exercises:
1) Print a few more interesting properties and read up in the specification what they mean.
*/
