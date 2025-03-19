#include "private.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

#include "ze_api_wrap.hpp"

#if 1
sycl::event launchOpenCLKernelOnlineLevelZero(sycl::queue &q, std::string source,
											  std::string func_name, std::vector<void *> &params,
											  sycl::event &dep_event, bool test_performance)
{
	return sycl::event();
}
#else
static bool get_env(std::string env)
{
	auto env_str = std::getenv(env.c_str());
	if (env_str && std::string("1") == env_str)
	{
		std::cout << "  == Get: " << env << " = 1" << std::endl;
		return true;
	}
	std::cout << "  == Get: " << env << " = 0" << std::endl;
	return false;
}

static std::string load_kernel(std::string kernel_fn)
{
	std::ifstream kernel_file(kernel_fn.c_str(), std::ios::in | std::ios::binary);
	if (kernel_file.is_open())
	{
		std::string ret;
		auto beg = kernel_file.tellg();
		kernel_file.seekg(0, std::ios::end);
		auto end = kernel_file.tellg();
		kernel_file.seekg(0, std::ios::beg);

		ret.resize((size_t)(end - beg));
		kernel_file.read(&ret[0], (size_t)(end - beg));

		return {std::move(ret)};
	}
	else
	{
		std::cout << "  == Fail: can't read kernle source: " << kernel_fn << std::endl;
		exit(0);
	}

	return std::string();
}

#include <level_zero/ze_api.h>
namespace fs = std::filesystem;
#include <filesystem>
static std::vector<uint8_t> buildOpenCL2SPIRV(const std::string &source, const std::string &func_name)
{
	std::string src_fn = std::tmpnam(nullptr);
	std::string out_fn = std::tmpnam(nullptr);
	FILE *f = fopen(src_fn.c_str(), "wb");
	fwrite(source.c_str(), source.length(), 1, f);
	fclose(f);

	std::string cmd = "ocloc -file " + src_fn + "  -device dg2 -output " + out_fn;
	system(cmd.c_str());

	std::string spirv_fn = out_fn + "_dg2.spv";
	std::ifstream spirv_file(spirv_fn, std::ios::binary);
	if (!spirv_file.is_open())
	{
		std::cout << "  == Fail: Can't open file: " << spirv_fn << std::endl;
		exit(0);
	}
	std::vector<uint8_t> spirv_binary((std::istreambuf_iterator<char>(spirv_file)), std::istreambuf_iterator<char>());
	std::cout << "  == Readed spirv kernel file: " << spirv_fn << std::endl;
	spirv_file.close();
	return spirv_binary;
}

sycl::event launchOpenCLKernelOnlineLevelZero(sycl::queue &q, std::string source,
											  std::string func_name, std::vector<void *> &params,
											  sycl::event &dep_event, bool test_performance)
{
	auto spv = buildOpenCL2SPIRV(source, func_name);

	ze_module_handle_t zeModule;
	ze_module_desc_t moduleDesc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
	ze_module_build_log_handle_t buildlog;
	moduleDesc.pNext = nullptr;
	moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
	moduleDesc.pInputModule = spv.data();
	moduleDesc.inputSize = spv.size();
	moduleDesc.pConstants = nullptr;
	moduleDesc.pBuildFlags = "";

	auto zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
		q.get_device());
	auto zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
		q.get_context());

	auto r = zeModuleCreate(zeContext, zeDevice, &moduleDesc, &zeModule, &buildlog);
	if (r != ZE_RESULT_SUCCESS)
	{
		size_t szLog = 0;
		zeModuleBuildLogGetString(buildlog, &szLog, nullptr);

		char *strLog = (char *)malloc(szLog);
		zeModuleBuildLogGetString(buildlog, &szLog, strLog);
		std::cout << "  == Fail: " << strLog << std::endl;

		free(strLog);
		return sycl::event();
	}

	ze_kernel_handle_t zeKernel;
	ze_kernel_desc_t kernelDesc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
	kernelDesc.pNext = nullptr;
	kernelDesc.flags = 0;
	kernelDesc.pKernelName = func_name.c_str();
	zeKernelCreate(zeModule, &kernelDesc, &zeKernel);

	for (size_t i = 0; i < params.size(); i++)
	{
		zeKernelSetArgumentValue(zeKernel, i, sizeof(void *), params[i]);
	}

	// sycl::nd_range ndr = sycl::nd_range{sycl::range{192, 14, 1}, sycl::range{192, 2, 1}};

	uint32_t groupSizeX = 1;
	uint32_t groupSizeY = 2u;
	uint32_t groupSizeZ = 192u;
	SUCCESS_OR_TERMINATE(zeKernelSetGroupSize(zeKernel, groupSizeX, groupSizeY, groupSizeZ));

	ze_group_count_t dispatchTraits;
	dispatchTraits.groupCountX = 1;
	dispatchTraits.groupCountY = 14u;
	dispatchTraits.groupCountZ = 192u;

	uint32_t computeQueueGroupOrdinal = 0;
	auto ret = get_cmd_queue_group_ordinal(zeDevice, computeQueueGroupOrdinal);
	std::cout << "getCmdQueue return " << (ret ? "Success" : "Fail") << std::endl;
	if (!ret)
	{
		exit(0);
	}
	std::cout << "Got computeQueueGroupOrdinal = " << computeQueueGroupOrdinal << std::endl;

	// Create a command list
	auto hCommandList = create_cmd_list(zeDevice, zeContext, computeQueueGroupOrdinal);
	std::cout << "Create command list: hCommandList = " << hCommandList << std::endl;

	// Create event pool
	auto hTSEventPool = create_event_pool_timestamp(zeContext);

	// Create a command queue
	auto hCommandQueue = create_cmd_queue(zeDevice, zeContext, computeQueueGroupOrdinal);
	std::cout << "Create command queue: hCommandQueue = " << hCommandQueue << std::endl;

	// Create event pool
	auto hEventPool = create_event_pool_host(zeContext);
	std::cout << "Create even pool: hEventPool = " << hEventPool << std::endl;

	// Create event
	auto hEvent = create_event_host(hEventPool);
	std::cout << "Create even: hEvent = " << hEvent << std::endl;

	for (auto i = 0; i < 150; i++)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		auto hTSEvent = create_event_timestamp(hTSEventPool);
		// std::cout << "Create even timestamp: hTSEvent = " << hTSEvent << std::endl;

		// Append a signal of a timestamp event into the command list after the kernel executes
		r = zeCommandListAppendLaunchKernel(hCommandList, zeKernel, &dispatchTraits, hTSEvent, 0, nullptr);
		CHECK_RET(r)

		// // Append a query of a timestamp event into the command list
		// r = zeCommandListAppendQueryKernelTimestamps(hCommandList, 1, &hTSEvent, tsResult, nullptr, hEvent, 1, &hTSEvent);
		// CHECK_RET(r)

		// Close list and submit for execution
		SUCCESS_OR_TERMINATE(zeCommandListClose(hCommandList));

		// Execute the command list with the signal
		// std::cout << "Command queue start to execute command list." << std::endl;
		r = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr);
		CHECK_RET(r)

		zeCommandListReset(hCommandList);

		// r = zeEventHostSynchronize(hEvent, 0);
		// CHECK_RET(r)

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "i = " << i << ", tm = " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " micro sec.\n";
	}

	SUCCESS_OR_TERMINATE(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint64_t>::max()));

	// // Wait on event to complete
	// r = zeEventHostSynchronize(hEvent, 0);
	// CHECK_RET(r)

	zeModuleBuildLogDestroy(buildlog);
	return sycl::event();
}
#endif