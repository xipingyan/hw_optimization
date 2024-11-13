// Reference:

#include <stdio.h>
#include <iostream>
#include <CL/opencl.hpp>
#include <stddef.h>
#include <stdint.h>

void dump_kernel_bin(cl::Program &program, std::string out_fn = "ocl_kernel_vec_add.bin")
{
	std::cout << "== Start to dump OCL kernel." << std::endl;
	cl_uint _n_device_num;
	cl_uint n_device_num = 1;
	if (clGetProgramInfo(program.get(), CL_PROGRAM_NUM_DEVICES,
						 sizeof(size_t), &_n_device_num, nullptr) != CL_SUCCESS ||
		_n_device_num != n_device_num)
	{
		printf("error: have %d devices, device query returns %d\n", int(n_device_num), int(_n_device_num));
		n_device_num = _n_device_num; // this fails on Intel MIC, I compiled for 1 device and get binaries for two!
	}
	std::cout << "    _n_device_num = " << _n_device_num << std::endl;
	std::cout << "    n_device_num = " << n_device_num << std::endl;

	size_t clDeviceBinarySize = 0;
	size_t paramValueSizeRet = 0;
	auto r = clGetProgramInfo(program.get(), CL_PROGRAM_BINARY_SIZES, sizeof(size_t), (void *)&clDeviceBinarySize, &paramValueSizeRet);
	std::cout << "  == Get CL_PROGRAM_BINARY_SIZES:" << std::endl;
	std::cout << "    r = " << r << std::endl;
	std::cout << "    clDeviceBinarySize = " << clDeviceBinarySize << std::endl;
	std::cout << "    paramValueSizeRet = " << paramValueSizeRet << std::endl;

	char **binaries = (char **)malloc(sizeof(char *) * n_device_num);
	for (int i = 0; i < n_device_num; i++)
		binaries[i] = (char *)malloc(clDeviceBinarySize);

	r = clGetProgramInfo(program.get(), CL_PROGRAM_BINARIES, clDeviceBinarySize, binaries, &paramValueSizeRet);
	std::cout << "  == Get CL_PROGRAM_BINARIES:" << std::endl;
	std::cout << "    r = " << std::hex << r << std::dec << std::endl;
	std::cout << "    paramValueSizeRet = " << paramValueSizeRet << std::endl;

	if (binaries)
	{
		for (int i = 0; i < n_device_num; i++)
		{
			FILE *pf = fopen(out_fn.c_str(), "wb");
			fwrite(binaries[i], sizeof(char), clDeviceBinarySize, pf);
			fclose(pf);
			free(binaries[i]);
		}
		free(binaries);
		binaries = nullptr;
	}
	std::cout << "== Finish dump OCL kernel." << std::endl;
}

int main()
{
	std::cout << "== Hello OpenCL(CPP)." << std::endl;
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

	// kernel calculates for each element C=A+B
	std::string kernel_code =
		"   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
		"       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
		"		printf(\"== kernel insdie: golbal_id=%zu \\n \", get_global_id(0));	"
		"   }                                                                               ";

	std::cout << "== Put kernel string to source." << std::endl;
	sources.push_back({kernel_code.c_str(), kernel_code.length()});

	std::cout << "== Construct program with source and context." << std::endl;
	cl::Program program(context, sources);
	if (program.build({default_device}) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

	// create buffers on the device
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * 10);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * 10);

	int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

	std::cout << "== Create command queue" << std::endl;
	// create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	// write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * 10, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * 10, B);

	// run the kernel
	// cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
	// simple_add(buffer_A, buffer_B, buffer_C);

	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel kernel_add = cl::Kernel(program, "simple_add");
	kernel_add.setArg(0, buffer_A);
	kernel_add.setArg(1, buffer_B);
	kernel_add.setArg(2, buffer_C);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(10), cl::NullRange);
	queue.finish();

	dump_kernel_bin(program);

	std::cout << "== Read result." << std::endl;
	int C[10];
	// read result C from the device to array C
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

	int expected[10];
	for (int i = 0; i < 10; i++)
	{
		expected[i] = A[i] + B[i];
	}

	std::cout << "== Start compare result with expected.\n";
	bool is_expected = true;
	for (int i = 0; i < 10; i++)
	{
		if (expected[i] != C[i])
		{
			std::cout << "Index: " << i << " Result " << C[i] << "!=" << " Expected " << expected[i] << std::endl;
			is_expected = false;
		}
	}

	std::cout << "== Result is_expected = " << is_expected << std::endl;
	std::cout << "== Done." << std::endl;
	return 0;
}