#include <iostream>
#include <cstring>
#include <level_zero/ze_api.h>

#include "common.hpp"

/*
Link: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#command-queues-and-command-lists
Command Lists: 
	They are mostly associated with Host threads for simultaneous construction.
	construction can occur independently of command queue submission
Command Queues:
	are mostly associated with physical device properties, such as the number of input streams.
	Command queues provide (near) zero-latency access to the device.

Host: Thread; device: Stream；
*/

bool getCmdQueue(ze_device_handle_t hDevice, uint32_t& computeQueueGroupOrdinal) {
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

int main()
{
	ze_driver_handle_t hDriver = nullptr;
	ze_device_handle_t hDevice = nullptr;
	auto r = get_device(hDriver, hDevice);
	if (!r) {
		DEBUG_LOG << "Can't find GPU devices." << std::endl;
		return 0;
	}
	std::cout << "Got hDriver = " << hDriver << ", hDevice = " << hDevice << std::endl;

	uint32_t computeQueueGroupOrdinal = 0;
	auto ret = getCmdQueue(hDevice, computeQueueGroupOrdinal);

	std::cout << "getCmdQueue return " << (ret ? "Success" : "Fail") << std::endl;
	if (ret)
		std::cout << "Got computeQueueGroupOrdinal = " << computeQueueGroupOrdinal << std::endl;

	std::cout << "Done." << std::endl;
	return 0;
}

/*
Exercises:
1) Print a few more interesting properties and read up in the specification what they mean.
*/
