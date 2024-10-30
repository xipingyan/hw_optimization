#include <iostream>
#include <cstring>
#include <level_zero/ze_api.h>

#include "common.hpp"
#include "ze_api_wrap.hpp"

/*
Contexts: 驱动使用的逻辑对象，managing all memory, command queues/lists, modules, synchronization objects, etc.
	1: 能被多个device使用；
API: zeContextCreate
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
	std::cout << "Get driver: " << hDriver << std::endl;

	// Create context(s)
	std::cout << "Create context. " << std::endl;
	ze_context_desc_t ctxtDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0 };
	ze_context_handle_t hContextA, hContextB;
	r = zeContextCreate(hDriver, &ctxtDesc, &hContextA);
	CHECK_RET(r);
	r = zeContextCreate(hDriver, &ctxtDesc, &hContextB);
	CHECK_RET(r);
	
	std::cout << "Alloc host memory. " << std::endl;
	ze_host_mem_alloc_desc_t hostCtxtDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, 0, 0};
	void* ptrA = nullptr;
	void* ptrB = nullptr;
	r = zeMemAllocHost(hContextA, &hostCtxtDesc, 80, 0, &ptrA); CHECK_RET(r);
	r = zeMemAllocHost(hContextB, &hostCtxtDesc, 88, 0, &ptrB); CHECK_RET(r);

	memcpy(ptrA, ptrB, 0xe); // ok

	ze_memory_allocation_properties_t props = { ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, 0};
	ze_device_handle_t hOutDevice = nullptr;
	r = zeMemGetAllocProperties(hContextA, ptrA, &props, &hDevice);
	CHECK_RET(r);
	std::cout << "props.id = " << props.id << ", props.pageSize = " << props.pageSize << ", hOutDevice=" << hOutDevice << std::endl;

	// Doc note: "illegal: Context A has no knowledge of ptrB"
	// But it works here, I don't know why???
	r = zeMemGetAllocProperties(hContextA, ptrB, &props, &hDevice);
	CHECK_RET(r);
	std::cout << "props.id = " << props.id << ", props.pageSize = " << props.pageSize << ", hOutDevice=" << hOutDevice << std::endl;

	r = zeMemFree(hContextA, ptrA); CHECK_RET(r);
	r = zeMemFree(hContextB, ptrB); CHECK_RET(r);

	std::cout << "Before zeContextDestroy, zeContextGetStatus = " << zeContextGetStatus(hContextA) << std::endl;
	r = zeContextDestroy(hContextA); CHECK_RET(r);
	r = zeContextDestroy(hContextB); CHECK_RET(r);

	// ??, why still return ZE_RESULT_SUCCESS?
	std::cout << "After zeContextDestroy, zeContextGetStatus = " << zeContextGetStatus(hContextA) << std::endl;

	std::cout << "Done." << std::endl;
	return 0;
}

/*
Exercises:
1) Print a few more interesting properties and read up in the specification what they mean.
*/
