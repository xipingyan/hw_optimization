#include <cstring>

#include "common.hpp"
#include "ze_api_wrap.hpp"

/*
Link: https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#barriers

Execution barriers:
Memory barriers:
Range-based Memory barriers:

*/

#define NS_IN_SEC 1000000000LL

long get_file_size(const char *filename) {
    FILE *fp = fopen(filename, "r");

    if (fp==NULL)
        return -1;

    if (fseek(fp, 0, SEEK_END) < 0) {
        fclose(fp);
        return -1;
    }

    long size = ftell(fp);
    // release the resources when not required
    fclose(fp);
    return size;
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

	// ============================================
	// Create kernel from IL(SPIR)
	ze_module_handle_t hModule;
	
	// ZE_MODULE_FORMAT_NATIVE, ZE_MODULE_FORMAT_FORCE_UINT32
	ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;
	size_t inputSize = 0;
	uint8_t *pInputModule = nullptr; 
	const char* pBuildFlags = "-ze-opt-disable";
	const char* fn = "../../../sycl_learn/CodeSamples/build/sycl_spir64.spv";
	inputSize = get_file_size(fn);
	FILE *f = fopen(fn, "rb");
	if (f==nullptr) {
		std::cout << "Can't open spir file." << std::endl;
		return 0;
	}
	pInputModule = (uint8_t*)malloc(inputSize);
	fclose(f);
	const ze_module_constants_t* pConstants = nullptr;
	ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC, 0, format, inputSize, pInputModule, pBuildFlags, pConstants};
	r = zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, nullptr);
	CHECK_RET(r)

	ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC, 0, ZE_KERNEL_FLAG_FORCE_RESIDENCY, "vec_add"};
	ze_kernel_handle_t hKernel;
	r = zeKernelCreate(hModule, &kernelDesc, &hKernel);
	CHECK_RET(r)

	// Append a signal of a timestamp event into the command list after the kernel executes
	ze_group_count_t launchFuncArgs = {1, 1, 1};
	r = zeCommandListAppendLaunchKernel(hCommandList, hKernel, &launchFuncArgs, hTSEvent, 0, nullptr);
	CHECK_RET(r)

	// Append a query of a timestamp event into the command list
	r = zeCommandListAppendQueryKernelTimestamps(hCommandList, 1, &hTSEvent, tsResult, nullptr, hEvent, 1, &hTSEvent);
	CHECK_RET(r)

	// Execute the command list with the signal
	std::cout << "Command queue start to execute command list." << std::endl;
	r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
	CHECK_RET(r)

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
	free(pInputModule);
	return 0;
}

/*
Exercises:
1) 
*/
