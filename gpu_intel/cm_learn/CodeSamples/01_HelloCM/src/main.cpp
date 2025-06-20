#include <limits>

#include "ze_api_wrap.hpp"
#include "my_common.hpp"

/************************************************
 * Convert cm kernle to spv
 * $ ../../llvm-project/build/install/bin/cmc -march=SKL ../01_HelloCM/src/hello_cm_kernel.cm -fcmocl -emit-spirv -o hello_cm_kernel.spv
 ************************************************/

bool spirv_cm_kernel(ze_device_handle_t hDevice, ze_context_handle_t hContext, ze_module_handle_t &hModule, ze_kernel_handle_t &hKernel)
{
	const char* fn = "./hello_cm_kernel.spv";
	DEBUG_LOG << "== Start to read SPIR-V kernel: " << fn << std::endl;
	auto spirBinFile = CKernelBinFile::createPtr(fn);
	std::cout << "  == kernel size: " << spirBinFile->_fileSize << std::endl;

	ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
	ze_module_build_log_handle_t buildlog;
	moduleDesc.pNext = nullptr;
	moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
	moduleDesc.pInputModule = spirBinFile->_pbuf;
	moduleDesc.inputSize = spirBinFile->_fileSize;
	moduleDesc.pConstants = nullptr;
	moduleDesc.pBuildFlags = "";
	auto r = zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, &buildlog);
	if (r != ZE_RESULT_SUCCESS) {
		size_t szLog = 0;
		zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

		char *strLog = (char *)malloc(szLog);
		zeModuleBuildLogGetString(buildlog, &szLog, strLog);
		std::cout << "  == Fail: " << strLog << std::endl;

		free(strLog);
		return false;
	}
	
	ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
	kernelDesc.pNext = nullptr;
	kernelDesc.flags = 0;
	kernelDesc.pKernelName = "hello_world";
	SUCCESS_OR_TERMINATE(zeKernelCreate(hModule, &kernelDesc, &hKernel));

	zeModuleBuildLogDestroy(buildlog);
	return true;
}

#define NS_IN_SEC 1000000000LL

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
	auto hCommandQueue = create_cmd_queue(hDevice, hContext, computeQueueGroupOrdinal);
	std::cout << "Create command queue: hCommandQueue = " << hCommandQueue << std::endl;

	// Create a command list
	auto hCommandList = create_cmd_list(hDevice, hContext, computeQueueGroupOrdinal);
	std::cout << "Create command list: hCommandList = " << hCommandList << std::endl;

	// Create event pool
	auto hEventPool = create_event_pool_host(hContext);
	std::cout << "Create even pool: hEventPool = " << hEventPool << std::endl;

	// Create event
	auto hEvent = create_event_host(hEventPool);
	std::cout << "Create even: hEvent = " << hEvent << std::endl;

	// Get timestamp frequency
	auto device_properties = get_properities(hDevice);
	const double timestampFreq = NS_IN_SEC / device_properties.timerResolution;
	const uint64_t timestampMaxValue = ~(-1L << device_properties.kernelTimestampValidBits);

	// Create event pool
	auto hTSEventPool = create_event_pool_timestamp(hContext);
	std::cout << "Create even pool timestamp: hTSEventPool = " << hEvent << std::endl;

	auto hTSEvent = create_event_timestamp(hTSEventPool);
	std::cout << "Create even timestamp: hTSEvent = " << hTSEvent << std::endl;

	// allocate memory for results
	ze_device_mem_alloc_desc_t tsResultDesc = {
		ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
		nullptr,
		0, // flags
		0  // ordinal
	};
	ze_kernel_timestamp_result_t* tsResult = nullptr;
	r = zeMemAllocDevice(hContext, &tsResultDesc, sizeof(ze_kernel_timestamp_result_t), sizeof(uint32_t), hDevice, reinterpret_cast<void**>(&tsResult));
	CHECK_RET(r)
	std::cout << "Alloc device memory: tsResult = " << tsResult << std::endl;

	ze_module_handle_t hModule;
	ze_kernel_handle_t hKernel;

	// input params:
	spirv_cm_kernel(hDevice, hContext, hModule, hKernel);

	ze_host_mem_alloc_desc_t host_desc = {};
	host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
	host_desc.flags = 0;
	host_desc.pNext = nullptr;

    int thread_width  = 8;
    int thread_height = 8;

	zeKernelSetArgumentValue(hKernel, 0, sizeof(thread_width), &thread_width);

	uint32_t groupSizeX = 32u;
	uint32_t groupSizeY = 1u;
	uint32_t groupSizeZ = 1u;
	SUCCESS_OR_TERMINATE(zeKernelSuggestGroupSize(hKernel, thread_height, 1U, 1U, &groupSizeX, &groupSizeY, &groupSizeZ));
	std::cout << "== suggest group: x=" << groupSizeX << ", y=" << groupSizeY << ", z=" << groupSizeZ << std::endl;
	SUCCESS_OR_TERMINATE(zeKernelSetGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ));

	ze_group_count_t dispatchTraits;
	dispatchTraits.groupCountX = thread_height / groupSizeX;
	dispatchTraits.groupCountY = 1u;
	dispatchTraits.groupCountZ = 1u;

	// Append a signal of a timestamp event into the command list after the kernel executes
	r = zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, hTSEvent, 0, nullptr);
	CHECK_RET(r)

	// Append a query of a timestamp event into the command list
	r = zeCommandListAppendQueryKernelTimestamps(hCommandList, 1, &hTSEvent, tsResult, nullptr, hEvent, 1, &hTSEvent);
	CHECK_RET(r)

	// Close list and submit for execution
    SUCCESS_OR_TERMINATE(zeCommandListClose(hCommandList));

	// Execute the command list with the signal
	std::cout << "== Command queue start to execute command list." << std::endl;
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
	CHECK_RET(r)

	std::cout << "== zeCommandQueueSynchronize." << std::endl;
	SUCCESS_OR_TERMINATE(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint64_t>::max()));

	// Wait on event to complete
	std::cout << "== zeEventHostSynchronize." << std::endl;
	r = zeEventHostSynchronize(hEvent, 0);
	CHECK_RET(r)

	// Calculation execution time(s) // Crash, I don't know why?
	// double globalTimeInNs = ( tsResult->global.kernelEnd >= tsResult->global.kernelStart )
	// 	? ( tsResult->global.kernelEnd - tsResult->global.kernelStart ) * timestampFreq
	// 	: (( timestampMaxValue - tsResult->global.kernelStart) + tsResult->global.kernelEnd + 1 ) * timestampFreq;

	// double contextTimeInNs = ( tsResult->context.kernelEnd >= tsResult->context.kernelStart )
	// 	? ( tsResult->context.kernelEnd - tsResult->context.kernelStart ) * timestampFreq
	// 	: (( timestampMaxValue - tsResult->context.kernelStart) + tsResult->context.kernelEnd + 1 ) * timestampFreq;

    // std::cout << "== globalTimeInNs = " << globalTimeInNs << std::endl;
	std::cout << "== Done." << std::endl;
	return 0;
}