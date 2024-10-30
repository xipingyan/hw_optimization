#include <iostream>
#include <cstring>
#include <level_zero/ze_api.h>

#include "common.hpp"
#include "ze_api_wrap.hpp"

/*
Link: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#synchronization-primitives

Fences:
	A fence is a heavyweight synchronization primitive used to communicate to the host that command list execution has completed.

Events:
	fine-grain
*/


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
	auto ret = get_cmd_queue_group_ordinal(hDevice, computeQueueGroupOrdinal);

	std::cout << "getCmdQueue return " << (ret ? "Success" : "Fail") << std::endl;
	if (!ret) {
		return EXIT_FAILURE;
	}
	std::cout << "Got computeQueueGroupOrdinal = " << computeQueueGroupOrdinal << std::endl;

	// Create context
	ze_context_handle_t hContext = create_context(hDriver);
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

	// Fence and Event:
#if 0
	// Create fence
	ze_fence_desc_t fenceDesc = {
		ZE_STRUCTURE_TYPE_FENCE_DESC,
		nullptr,
		0 // flags
	};
	ze_fence_handle_t hFence;
	r = zeFenceCreate(hCommandQueue, &fenceDesc, &hFence);
	CHECK_RET(r)

	// Execute a command list with a signal of the fence
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, hFence);
	CHECK_RET(r)

	// Wait for fence to be signaled
	r = zeFenceHostSynchronize(hFence, UINT32_MAX);
	CHECK_RET(r)

	r = zeFenceReset(hFence);
	CHECK_RET(r)
#else
	// Create event pool
	ze_event_pool_desc_t eventPoolDesc = {
		ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
		nullptr,
		ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
		1 // count
	};
	ze_event_pool_handle_t hEventPool;
	zeEventPoolCreate(hContext, &eventPoolDesc, 0, nullptr, &hEventPool);

	ze_event_desc_t eventDesc = {
		ZE_STRUCTURE_TYPE_EVENT_DESC,
		nullptr,
		0, // index
		0, // no additional memory/cache coherency required on signal
		ZE_EVENT_SCOPE_FLAG_HOST  // ensure memory coherency across device and Host after event completes
	};
	ze_event_handle_t hEvent;
	r = zeEventCreate(hEventPool, &eventDesc, &hEvent);
	CHECK_RET(r);

	// Append a signal of an event into the command list after the kernel executes
	//r = zeCommandListAppendLaunchKernel(hCommandList, hKernel1, &launchArgs, hEvent, 0, nullptr);
	//CHECK_RET(r);

	// Execute the command list with the signal
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
	CHECK_RET(r);

	// Wait on event to complete
	r = zeEventHostSynchronize(hEvent, 0);
	CHECK_RET(r);
#endif

	std::cout << "Done." << std::endl;
	return 0;
}

/*
Exercises:
1) Print a few more interesting properties and read up in the specification what they mean.
*/
