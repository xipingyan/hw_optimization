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
	if (!ret) {
		return EXIT_FAILURE;
	}
	std::cout << "Got computeQueueGroupOrdinal = " << computeQueueGroupOrdinal << std::endl;

	// Create context	
	ze_context_handle_t hContext;
	ze_context_desc_t ctxtDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0 };
	r = zeContextCreate(hDriver, &ctxtDesc, &hContext);
	CHECK_RET(r)
	std::cout << "Create context: hContext = " << hContext << std::endl;

	// Create a command queue
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
	r = zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue);
	CHECK_RET(r)
	std::cout << "Create command queue: hCommandQueue = " << hCommandQueue << std::endl;

	// Create a command list
	ze_command_list_desc_t commandListDesc = {
		ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
		nullptr,
		computeQueueGroupOrdinal,
		0 // flags
	};
	ze_command_list_handle_t hCommandList;
	zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList);
	CHECK_RET(r)
	std::cout << "Create command list: hCommandList = " << hCommandList << std::endl;

	// finished appending commands (typically done on another thread)
	r = zeCommandListClose(hCommandList);
	CHECK_RET(r)
	std::cout << "Close command list: hCommandList = " << hCommandList << std::endl;

	// Execute command list in command queue
	std::cout << "Execute command list in command queue" << std::endl;
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
	CHECK_RET(r)

	// synchronize host and device
	std::cout << "Synchronize host and device" << std::endl;
	r = zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX);
	CHECK_RET(r)

	// Reset (recycle) command list for new commands
	std::cout << "Reset command list for new commands" << std::endl;
	r = zeCommandListReset(hCommandList);
	CHECK_RET(r)

	// Create an immediate command list
	// ze_command_queue_desc_t commandQueueDesc = {
	// 	ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
	// 	nullptr,
	// 	computeQueueGroupOrdinal,
	// 	0, // index
	// 	0, // flags
	// 	ZE_COMMAND_QUEUE_MODE_DEFAULT,
	// 	ZE_COMMAND_QUEUE_PRIORITY_NORMAL
	// };
	r = zeCommandListCreateImmediate(hContext, hDevice, &commandQueueDesc, &hCommandList);
	std::cout << "Create an immediate command list: hCommandList = " << hCommandList << std::endl;

	// Immediately submit a kernel to the device
	//zeCommandListAppendLaunchKernel(hCommandList, hKernel, &launchArgs, nullptr, 0, nullptr);

	std::cout << "Done." << std::endl;
	return 0;
}

/*
Exercises:
1) Print a few more interesting properties and read up in the specification what they mean.
*/
