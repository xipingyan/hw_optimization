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

	// ============================================
	// Create kernel from IL(SPIR)
	// Refer: https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-level-zero-interface.html
	// https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_world_gpu.cpp

	// input params:
	size_t vec_sz = 10000u;
	std::vector<float> vec_a(vec_sz), vec_b(vec_sz);
	for (size_t i = 0; i < vec_sz; i++) {
		vec_a[i] = i;
		vec_b[i] = i;
	}

	ze_module_handle_t hModule;	
	const char* fn = "../../../sycl_learn/CodeSamples/build/sycl_spir64.spv";
	auto spirBinFile = CSpirKernelBinFile::createPtr(fn);
	const ze_module_constants_t* pConstants = nullptr;
	ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
	ze_module_build_log_handle_t buildlog;
	moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
	moduleDesc.pInputModule = spirBinFile->_pbuf;
	moduleDesc.inputSize = spirBinFile->_fileSize;
	moduleDesc.pBuildFlags = "";
	r = zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, &buildlog);
	if (r != ZE_RESULT_SUCCESS) {
		size_t szLog = 0;
		zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

		char *strLog = (char *)malloc(szLog);
		zeModuleBuildLogGetString(buildlog, &szLog, strLog);
		std::cout << "== Fail: " << strLog << std::endl;

		free(strLog);
		// SUCCESS_OR_TERMINATE(zeModuleBuildLogDestroy(buildlog));
		std::cerr << "\nZello World Gpu Results validation FAILED. Module creation error."
					<< std::endl;
		return 0;
	}

	ze_kernel_handle_t hKernel;
	ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
	kernelDesc.pKernelName = "vec_add";
	SUCCESS_OR_TERMINATE(zeKernelCreate(hModule, &kernelDesc, &hKernel));

	uint32_t groupSizeX = 32u;
	uint32_t groupSizeY = 1u;
	uint32_t groupSizeZ = 1u;
	SUCCESS_OR_TERMINATE(zeKernelSuggestGroupSize(hKernel, vec_sz, 1U, 1U, &groupSizeX, &groupSizeY, &groupSizeZ));
	std::cout << "== suggest group: x=" << groupSizeX << ", y=" << groupSizeY << ", z=" << groupSizeZ << std::endl;
	SUCCESS_OR_TERMINATE(zeKernelSetGroupSize(hKernel, groupSizeX, groupSizeY, groupSizeZ));

	uint32_t offset = 0;
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 1, sizeof(dstBuffer), &dstBuffer));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 0, sizeof(srcBuffer), &srcBuffer));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 2, sizeof(uint32_t), &offset));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 3, sizeof(uint32_t), &offset));
	// SUCCESS_OR_TERMINATE(zeKernelSetArgumentValue(hKernel, 4, sizeof(uint32_t), &offset));

	ze_group_count_t dispatchTraits;
	dispatchTraits.groupCountX = vec_sz / groupSizeX;
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
	
	std::cout << "Done." << std::endl;

	zeModuleBuildLogDestroy(buildlog);
	return 0;
}

/*
Exercises:
1) 
*/
