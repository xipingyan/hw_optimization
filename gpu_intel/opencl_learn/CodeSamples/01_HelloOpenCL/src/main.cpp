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

cl::Device get_gpu_device() {
	// get all platforms (drivers)
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0)
	{
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	size_t selected_platform = -1;
	for (size_t i = 0; i < all_platforms.size(); i++)
	{
		std::string platname = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
		if (platname.find("Graphics") != std::string::npos)
		{
			selected_platform = i;
			break;
		}
	}
	if (selected_platform == -1)
	{
		std::cout << " No GPU platforms is found. Check OpenCL installation!\n";
		exit(1);
	}

	cl::Platform default_platform = all_platforms[selected_platform];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	if (all_devices.size() == 0)
	{
		std::cout << " No GPU device is found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[0];

	return default_device;
}

void run_kernel(cl::CommandQueue &queue, cl::Context &context, cl::Kernel kernel_add, std::vector<std::vector<int>> &inputs, std::vector<int> &output)
{
	size_t num = inputs[0].size();
	if (inputs[0].size() != output.size())
	{
		std::cout << "  Fail: input output size diff." << std::endl;
		exit(0);
	}

	// create buffers on the device
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * num);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * num);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * num);

	// write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * num, inputs[0].data());
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * num, inputs[1].data());

	kernel_add.setArg(0, buffer_A);
	kernel_add.setArg(1, buffer_B);
	kernel_add.setArg(2, buffer_C);

	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(num), cl::NullRange);
	queue.finish();

	// dump_kernel_bin(program);

	std::cout << "  == Start to copy output from device to host" << std::endl;
	// Copy output from device to host
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * num, output.data());
}

template <typename T>
bool is_close(const std::vector<T> &vec1, const std::vector<T> &vec2)
{
	// 1. Check if the sizes are different
	if (vec1.size() != vec2.size())
	{
		return false;
	}

	// 2. Iterate through elements and compare them
	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (vec1[i] != vec2[i])
		{
			return false; // Found a differing element
		}
	}

	// If we reach here, sizes are the same and all elements are equal
	return true;
}
template <typename T>
std::vector<T> addVectors(const std::vector<T>& vec1, const std::vector<T>& vec2) {
    // Check if vectors have the same size
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Vectors must have the same size to be added element-wise.");
    }

    // Create a new vector to store the result
    std::vector<T> result_vec;
    result_vec.reserve(vec1.size()); // Pre-allocate memory for efficiency

    // Iterate through elements and add them
    for (size_t i = 0; i < vec1.size(); ++i) {
        result_vec.push_back(vec1[i] + vec2[i]);
    }

    return result_vec;
}

int main()
{
	std::cout << "== Hello OpenCL(CPP)." << std::endl;

	auto default_device = get_gpu_device();
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	std::cout << "== Create context" << std::endl;
	cl::Context context({default_device});

	std::cout << "== Create Sources" << std::endl;
	cl::Program::Sources sources;

	// kernel calculates for each element C=A+B
	std::string kernel_code =
		"   void kernel simple_add(global const int* A, global const int* B, global int* C){       "
		"       C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)];                 "
		"		printf(\"== kernel inside: golbal_id=%zu \\n \", get_global_id(0));	"
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

	// Construct kernel 1
	cl::vector<cl::Kernel> kernels;
	program.createKernels(&kernels);
	if (kernels.size() > 0)
	{
		auto kernel_name = kernels[0].getInfo<CL_KERNEL_FUNCTION_NAME>();
		std::cout << "  == Get kernel function name from  = " << kernel_name << std::endl;
	}

	std::cout << "== Create command queue" << std::endl;
	// create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	// run the kernel
	// cl::KernelFunctor simple_add(cl::Kernel(program, "simple_add"), queue, cl::NullRange, cl::NDRange(10), cl::NullRange);
	// simple_add(buffer_A, buffer_B, buffer_C);

	std::cout << "== Create Kernel with program and run." << std::endl;
	// alternative way to run the kernel
	cl::Kernel kernel_add = cl::Kernel(program, "simple_add");    // Construct kernel 2

	auto kernel_name = kernel_add.getInfo<CL_KERNEL_FUNCTION_NAME>();
	std::cout << "== Test get kernel name from cl::Kernel, kernel_name = " << kernel_name << std::endl;
	// auto old_kernel_program = kernel_add.getInfo<CL_KERNEL_PROGRAM>();

	// ==================
	std::cout << "== Start to run." << std::endl;
	std::vector<std::vector<int>> input_a_b = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
											   {0, 1, 2, 0, 1, 2, 0, 1, 2, 0}};
	std::vector<int> output(10);
	run_kernel(queue, context, kernel_add, input_a_b, output);

	auto expected_ab = addVectors<int>(input_a_b[0], input_a_b[1]);
	std::cout << "  == Result CPU VS GPU = " << is_close<int>(expected_ab, output) << std::endl;

	// ==================
	std::cout << "== Start to run." << std::endl;
	std::vector<std::vector<int>> input_a_b2 = {{0, 1, 2, 3, 4},
											   {0, 1, 2, 0, 1}};
	std::vector<int> output2(5);
	run_kernel(queue, context, kernel_add, input_a_b2, output2);

	auto expected_ab2 = addVectors<int>(input_a_b2[0], input_a_b2[1]);
	std::cout << "  == Result CPU VS GPU = " << is_close<int>(expected_ab2, output2) << std::endl;

	std::cout << "== Done." << std::endl;
	return 0;
}