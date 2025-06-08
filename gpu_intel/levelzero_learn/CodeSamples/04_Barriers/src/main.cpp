#include <cstring>
#include <limits>

#include "my_common.hpp"
#include "ze_api_wrap.hpp"

/*
Link: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#barriers

Execution barriers:
Memory barriers:
Range-based Memory barriers:

*/

#define NS_IN_SEC 1000000000LL

bool kernel_spir(ze_device_handle_t hDevice, ze_context_handle_t hContext, ze_kernel_handle_t& hKernel) {
	std::cout << "== Start read SPIR-V kernel." << std::endl;

	// ============================================
	// Create kernel from IL(SPIR)
	// Refer: https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html
	// https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_world_gpu.cpp

	ze_module_handle_t hModule;	
	const char* fn = "../../../sycl_learn/CodeSamples/build/sycl_spir64.spv";
	auto spirBinFile = CKernelBinFile::createPtr(fn);
	const ze_module_constants_t* pConstants = nullptr;
	ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
	ze_module_build_log_handle_t buildlog;
	moduleDesc.pNext = nullptr;
	moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
	moduleDesc.pInputModule = spirBinFile->_pbuf;
	moduleDesc.inputSize = spirBinFile->_fileSize;
	moduleDesc.pBuildFlags = "";
	std::cout << "  == Start create module." << std::endl;
	auto r = zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, &buildlog);
	if (r != ZE_RESULT_SUCCESS) {
		size_t szLog = 0;
		r = zeModuleBuildLogGetString(buildlog, &szLog, nullptr);
		std::cout << "  szLog = " << szLog << ", r = " << r << std::endl;

		char *strLog = (char *)malloc(szLog);
		r = zeModuleBuildLogGetString(buildlog, &szLog, strLog);
		std::cout << "  == strLog=" << strLog << ", r=" << r << std::endl;

		free(strLog);
		// SUCCESS_OR_TERMINATE(zeModuleBuildLogDestroy(buildlog));
		return false;
	}

	std::cout << "  == Start create kernel." << std::endl;
	ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
	// Note: find kernel name based on OpEntryPoint, open spv file and check via https://www.khronos.org/spir/visualizer/
	kernelDesc.pKernelName = "_ZTSN4sycl3_V16detail18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZ7vec_addRSt6vectorIfSaIfEES8_RNS0_5queueEEUlNS0_2idILi1EEEE_EE";
	
	r = zeKernelCreate(hModule, &kernelDesc, &hKernel);
	if (r != ZE_RESULT_SUCCESS) {
		std::cout << "    r = " << std::hex << r << std::dec << (r == 0x78000011 ? " ERROR_INVALID_KERNEL_NAME" : "") << std::endl;
		size_t szLog = 0;
		r = zeModuleBuildLogGetString(buildlog, &szLog, nullptr);
		std::cout << "    szLog = " << szLog << ", r = " << r << std::endl;

		char *strLog = (char *)malloc(szLog);
		r = zeModuleBuildLogGetString(buildlog, &szLog, strLog);
		std::cout << "    strLog=" << strLog << ", r=" << r << std::endl;

		free(strLog);
		SUCCESS_OR_TERMINATE(zeModuleBuildLogDestroy(buildlog));
		return false;
	}

	zeModuleBuildLogDestroy(buildlog);
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

	// input params:
	auto allocSize = 1;
	void *buffer_A = nullptr;
	void *buffer_B = nullptr;
	void *buffer_C = nullptr;
	ze_device_mem_alloc_desc_t deviceDesc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    deviceDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    deviceDesc.ordinal = 0;
	ze_host_mem_alloc_desc_t hostDesc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    hostDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    SUCCESS_OR_TERMINATE(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, allocSize * sizeof(int), 1, hDevice, &buffer_A));
    SUCCESS_OR_TERMINATE(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, allocSize * sizeof(int), 1, hDevice, &buffer_B));
	SUCCESS_OR_TERMINATE(zeMemAllocShared(hContext, &deviceDesc, &hostDesc, allocSize * sizeof(int), 1, hDevice, &buffer_C));

	for (size_t i = 0; i < allocSize; i++) {
		((float*)buffer_A)[i] = i;
		((float*)buffer_B)[i] = i;
	}

	ze_kernel_handle_t hKernel;
	kernel_spir(hDevice, hContext, hKernel);

	uint32_t groupSizeX = 32u;
	uint32_t groupSizeY = 1u;
	uint32_t groupSizeZ = 1u;
	SUCCESS_OR_TERMINATE(zeKernelSuggestGroupSize(hKernel, allocSize, 1U, 1U, &groupSizeX, &groupSizeY, &groupSizeZ));
	std::cout << "== suggest group: x=" << groupSizeX << ", y=" << groupSizeY << ", z=" << groupSizeZ << std::endl;
	SUCCESS_OR_TERMINATE(zeKernelSetGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ));

	uint32_t offset = 0;
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 0, sizeof(float) * allocSize, &buffer_A));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 1, sizeof(float) * allocSize, &buffer_B));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 2, sizeof(float) * allocSize, &buffer_C));

	ze_group_count_t dispatchTraits;
	dispatchTraits.groupCountX = allocSize / groupSizeX;
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
	std::cout << "Command queue start to execute command list." << std::endl;
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
	CHECK_RET(r)

	SUCCESS_OR_TERMINATE(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint64_t>::max()));

	// Wait on event to complete
	r = zeEventHostSynchronize(hEvent, 0);
	CHECK_RET(r)

	// Calculation execution time(s)
	double globalTimeInNs = ( tsResult->global.kernelEnd >= tsResult->global.kernelStart )
		? ( tsResult->global.kernelEnd - tsResult->global.kernelStart ) * timestampFreq
		: (( timestampMaxValue - tsResult->global.kernelStart) + tsResult->global.kernelEnd + 1 ) * timestampFreq;

	double contextTimeInNs = ( tsResult->context.kernelEnd >= tsResult->context.kernelStart )
		? ( tsResult->context.kernelEnd - tsResult->context.kernelStart ) * timestampFreq
		: (( timestampMaxValue - tsResult->context.kernelStart) + tsResult->context.kernelEnd + 1 ) * timestampFreq;
	
	std::cout << "== Infer result: " << std::endl;
	for (size_t i = 0; i < 10; i++) {
		std::cout << "  buffer_C[" << i << "] = " << ((float*)buffer_C)[i] << std::endl;
	}
	std::cout << "Done." << std::endl;

	return 0;
}

/*
Exercises:
1) 
*/
