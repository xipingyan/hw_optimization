#pragma once
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

inline bool get_device(ze_driver_handle_t& hDriver, ze_device_handle_t& hDevice) {
	// Initialize the driver
	auto r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
	CHECK_RET(r)

	// Discover all the driver instances
	uint32_t driverCount = 0;
	r = zeDriverGet(&driverCount, nullptr);
	CHECK_RET(r)

	ze_driver_handle_t* allDrivers = (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
	r = zeDriverGet(&driverCount, allDrivers);
	CHECK_RET(r)

	// Find a driver instance with a GPU device
	// ze_driver_handle_t hDriver = nullptr;
	// ze_device_handle_t hDevice = nullptr;
	for(uint32_t i = 0; i < driverCount; ++i) {
		uint32_t deviceCount = 0;
		r = zeDeviceGet(allDrivers[i], &deviceCount, nullptr);
		CHECK_RET(r)

		ze_device_handle_t* allDevices = (ze_device_handle_t*)malloc(deviceCount * sizeof(ze_device_handle_t));
		r = zeDeviceGet(allDrivers[i], &deviceCount, allDevices);
		CHECK_RET(r)

		for(uint32_t d = 0; d < deviceCount; ++d) {
			ze_device_properties_t device_properties {};
			device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
			r = zeDeviceGetProperties(allDevices[d], &device_properties);
			CHECK_RET(r)

			if(ZE_DEVICE_TYPE_GPU == device_properties.type) {
				hDriver = allDrivers[i];
				hDevice = allDevices[d];
				std::cout << "== Got device: " << device_properties.name << std::endl;
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

ze_device_properties_t get_properities(ze_device_handle_t hDevice) {
	ze_device_properties_t device_properties {};
	device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
	auto r = zeDeviceGetProperties(hDevice, &device_properties);
	CHECK_RET(r)
	return device_properties;
}

ze_context_handle_t create_context(ze_driver_handle_t hDriver) {
	ze_context_handle_t hContext;
	ze_context_desc_t ctxtDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0 };
	auto r = zeContextCreate(hDriver, &ctxtDesc, &hContext);
	CHECK_RET(r)
	return hContext;
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

ze_command_queue_handle_t create_cmd_queue(ze_device_handle_t hDevice, ze_context_handle_t hContext, uint32_t computeQueueGroupOrdinal) {
	ze_command_queue_desc_t commandQueueDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
		nullptr,
		computeQueueGroupOrdinal,
		0, // index
		0, // flags
		ZE_COMMAND_QUEUE_MODE_DEFAULT,
		ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	};
	ze_command_queue_handle_t hCommandQueue;
	auto r = zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue);
	CHECK_RET(r)
	return hCommandQueue;
}

ze_command_list_handle_t create_cmd_list(ze_device_handle_t hDevice, ze_context_handle_t hContext, uint32_t computeQueueGroupOrdinal) {
	ze_command_list_desc_t commandListDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
		nullptr,
		computeQueueGroupOrdinal,
		0 // flags
	};
	ze_command_list_handle_t hCommandList;
	auto r = zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList);
	CHECK_RET(r)
	return hCommandList;
}

ze_event_pool_handle_t create_event_pool_host(ze_context_handle_t hContext) {
	ze_event_pool_desc_t eventPoolDesc = {
		ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
		nullptr,
		ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
		1 // count
	};
	ze_event_pool_handle_t hEventPool;
	auto r = zeEventPoolCreate(hContext, &eventPoolDesc, 0, nullptr, &hEventPool);
	CHECK_RET(r)
	return hEventPool;
}

ze_event_handle_t create_event_host(ze_event_pool_handle_t hEventPool) {
	ze_event_desc_t eventDesc = {
		ZE_STRUCTURE_TYPE_EVENT_DESC,
		nullptr,
		0, // index
		0, // no additional memory/cache coherency required on signal
		ZE_EVENT_SCOPE_FLAG_HOST  // ensure memory coherency across device and Host after event completes
	};
	ze_event_handle_t hEvent;
	auto r = zeEventCreate(hEventPool, &eventDesc, &hEvent);
	CHECK_RET(r);
	return hEvent;
}

ze_event_pool_handle_t create_event_pool_timestamp(ze_context_handle_t hContext) {
	ze_event_pool_desc_t eventPoolDesc = {
		ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
		nullptr,
		ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP, // all events in pool are kernel timestamps
		1 // count
	};
	ze_event_pool_handle_t hTSEventPool;
	auto r = zeEventPoolCreate(hContext, &eventPoolDesc, 0, nullptr, &hTSEventPool);
	CHECK_RET(r)
	return hTSEventPool;
}

ze_event_handle_t create_event_timestamp(ze_event_pool_handle_t hEventPool) {
	ze_event_desc_t eventDesc = {
		ZE_STRUCTURE_TYPE_EVENT_DESC,
		nullptr,
		0, // index
		0, // no additional memory/cache coherency required on signal
		0  //  no additional memory/cache coherency required on wait
	};
	ze_event_handle_t hTSEvent;
	auto r = zeEventCreate(hEventPool, &eventDesc, &hTSEvent);
	CHECK_RET(r);
	return hTSEvent;
}