// Reference:

#include <stdio.h>
#include <iostream>
#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>
#include <fstream>

#include "private.hpp"

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
	return std::string();
}

int main()
{
	std::cout << "== 02_Rope_kernel_opencl." << std::endl;
	// get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0)
	{
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::cout << "== Create context" << std::endl;
	cl::Context context({default_device});

	std::cout << "== Create Sources" << std::endl;
	cl::Program::Sources sources;

	// kernel
	auto kernel_fn = "../../../sycl_learn/CodeSamples/02_sycl_ocl_interoperate/src/kernel_rope_ref/SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl";
	std::cout << "== load kernel: " << kernel_fn << std::endl;
	std::string kernel_code = load_kernel(kernel_fn);

	std::cout << "== Put kernel string to source." << std::endl;
	sources.push_back({kernel_code.c_str(), kernel_code.length()});

	std::cout << "== Construct program with source and context." << std::endl;
	cl::Program program(context, sources);
	if (program.build({default_device}) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

	// Construct kernel 1
	cl::vector<cl::Kernel> kernels;
	program.createKernels(&kernels);
	for (auto func_name : kernels)
	{
		auto kernel_name = func_name.getInfo<CL_KERNEL_FUNCTION_NAME>();
		std::cout << "  == Get kernel function name from  = " << kernel_name << std::endl;
	}

	DumpData input_0;
	input_0.data = std::vector<float>({1, 6, 1, 1, 1, 1, 14, 64, 0, 4, 1, 1, 1, 1, 1, 1, 6, 64, 1, 1, 1, 1, 1, 1, 6, 64, 1, 14, 1, 1, 1, 1, 6, 64});
	input_0.format = "bfyx";
	input_0.shape = {34};

	std::string root_path = "../../../sycl_learn/CodeSamples/02_sycl_ocl_interoperate/src/kernel_rope_ref/";
	auto input_1 = load_dump_data(root_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src0.txt");
	auto input_2 = load_dump_data(root_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src1.txt");
	auto input_3 = load_dump_data(root_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_src2.txt");
	auto output_expected = load_dump_data(root_path + "program1_network1_0_rope___module.model.layers.1.self_attn_aten__add_Add_dst0.txt");

	// create buffers on the device
	cl::Buffer inp_buf_0(context, CL_MEM_READ_ONLY, sizeof(int) * input_0.data.size());
	cl::Buffer inp_buf_1(context, CL_MEM_READ_ONLY, sizeof(cl_half16) * input_1.data.size());
	cl::Buffer inp_buf_2(context, CL_MEM_READ_ONLY, sizeof(cl_half16) * input_2.data.size());
	cl::Buffer inp_buf_3(context, CL_MEM_READ_ONLY, sizeof(cl_half16) * input_3.data.size());
	cl::Buffer out_buf_0(context, CL_MEM_READ_WRITE, sizeof(cl_half16) * output_expected.data.size());

	std::cout << "== Create command queue" << std::endl;
	// create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	// write arrays A and B to the device
	auto buf_0 = input_0.to_int();
	auto buf_1 = input_1.to_half();
	auto buf_2 = input_2.to_half();
	auto buf_3 = input_3.to_half();
	
	std::cout << "== enqueueWriteBuffer" << std::endl;
	queue.enqueueWriteBuffer(inp_buf_0, CL_TRUE, 0, sizeof(int) * input_0.data.size(), buf_0);
	queue.enqueueWriteBuffer(inp_buf_1, CL_TRUE, 0, sizeof(ushort) * input_1.data.size(), buf_1);
	queue.enqueueWriteBuffer(inp_buf_2, CL_TRUE, 0, sizeof(ushort) * input_2.data.size(), buf_2);
	queue.enqueueWriteBuffer(inp_buf_3, CL_TRUE, 0, sizeof(ushort) * input_3.data.size(), buf_3);
	
	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel kernel_add = cl::Kernel(program, "rope_ref_11982042700243959200_0_0__sa");    // Construct kernel 2
	kernel_add.setArg(0, inp_buf_0);
	kernel_add.setArg(1, inp_buf_1);
	kernel_add.setArg(2, inp_buf_2);
	kernel_add.setArg(3, inp_buf_3);
	kernel_add.setArg(4, out_buf_0);

	auto kernel_name = kernel_add.getInfo<CL_KERNEL_FUNCTION_NAME>();
	std::cout << "== Test get kernel name from cl::Kernel, kernel_name = " << kernel_name << std::endl;
	// auto old_kernel_program = kernel_add.getInfo<CL_KERNEL_PROGRAM>()
	// queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(192, 14, 1),  cl::NDRange(192, 2, 1));
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(1, 14, 192),  cl::NDRange(1, 2, 192));
	queue.finish();

	std::cout << "== Read result." << std::endl;
	int C[10];
	// read result C from the device to array C
	ushort* output = (ushort*)malloc(sizeof(ushort) * output_expected.data.size());
	queue.enqueueReadBuffer(out_buf_0, CL_TRUE, 0, sizeof(ushort) * output_expected.data.size(), output);

	std::cout << "== Start compare result with expected.\n";
	bool is_expected = true;
	for (int i = 0; i < output_expected.data.size(); i++)
	{
		if (std::to_string(output_expected.data[i]) != std::to_string(half_to_float(output[i])))
		{
			std::cout << "Index: " << i << " Result " << half_to_float(output[i]) << "!=" << " Expected " << output_expected.data[i] << std::endl;
			is_expected = false;
		}
	}

	// std::cout << "== Result is_expected = " << is_expected << std::endl;
	std::cout << "== Done." << std::endl;
	return 0;
}