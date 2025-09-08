#pragma once
#include <CL/opencl.hpp>
#include <iostream>
#include <map>
struct MyDevInfo
{
	size_t device_max_compute_unit = 0;	 // CL_DEVICE_MAX_COMPUTE_UNITS
	size_t device_max_group_size = 0;	 // lws不能超过这个值，CL_DEVICE_MAX_WORK_GROUP_SIZE
	size_t device_max_work_item_size[3]; // CL_DEVICE_MAX_WORK_ITEM_SIZES
};

// lws 比较喜欢这个值的倍数。 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
inline size_t get_kernel_perferred_workgroup_size_multiple(cl::Kernel kernel, cl::Device device)
{
	size_t kernel_perferred_workgroup_size_multiple = 0;
	auto err = clGetKernelWorkGroupInfo(
		kernel.get(),
		device.get(),
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(size_t),
		&kernel_perferred_workgroup_size_multiple,
		NULL);
	if (err == CL_SUCCESS)
	{
		// std::cout << "  Kernel preferred work group size multiple: " << kernel_perferred_workgroup_size_multiple << std::endl;
	}
	else
	{
		std::cout << "  Error: get CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, err = " << err << std::endl;
	}
	return kernel_perferred_workgroup_size_multiple;
}

inline cl::Device get_gpu_device()
{
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

class CMyTest
{
	std::string _kernel_fn = "../04_array_max/src/array_max_kernel.cl";
	std::string _kernel_entry = "get_array_max";

	cl::Device default_device = get_gpu_device();
	cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;
	std::shared_ptr<cl::CommandQueue> queue = nullptr;
	cl::Kernel defalt_kernel;
	std::map<std::string, cl::Kernel> _kernels;

public:
	CMyTest(const std::string kernel_entry, const std::string kernel_fn) : _kernel_fn(kernel_fn), _kernel_entry(kernel_entry)
	{
		default_device = get_gpu_device();
		context = cl::Context({default_device});

		std::cout << "== Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

		std::string kernel_code = load_kernel_source_codes(kernel_fn);
		if (kernel_code.empty())
		{
			std::cout << "  == Fail: can't load: " << kernel_fn.c_str() << std::endl;
			exit(0);
		}

		std::cout << "== Put kernel string to source." << std::endl;
		sources.push_back({kernel_code.c_str(), kernel_code.length()});

		std::cout << "== Construct program with source and context." << std::endl;
		program = cl::Program(context, sources);
		if (program.build({default_device}) != CL_SUCCESS)
		{
			std::cout << "  == Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
			exit(1);
		}

		// Construct kernel 1
		cl::vector<cl::Kernel> kernels;
		program.createKernels(&kernels);
		for (int k = 0; k < kernels.size(); k++)
		{
			auto kernel_name = kernels[k].getInfo<CL_KERNEL_FUNCTION_NAME>();
			std::cout << "  == kernels[" << k << "]  = " << kernel_name << std::endl;
		}

		std::cout << "== Create command queue" << std::endl;
		// create queue to which we will push commands for the device.
		queue = std::make_shared<cl::CommandQueue>(context, default_device);

		std::cout << "== Create Kernel with program and run." << std::endl;
		// alternative way to run the kernel
		defalt_kernel = cl::Kernel(program, kernel_entry.c_str());
		auto kernel_name = defalt_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
		std::cout << "  == Crurrent default kernel name = " << kernel_name << std::endl;
	}

	std::shared_ptr<cl::CommandQueue> get_queue() { return queue; }
	cl::Context get_context() { return context; }
	cl::Device get_device() { return default_device; }
	cl::Kernel get_kernel() { return defalt_kernel; }
	cl::Kernel get_kernel(std::string entry)
	{
		try
		{
			return _kernels.at(entry);
		}
		catch (const std::out_of_range &e)
		{
			std::cout << "  Warning: no exist kernel: " << entry << ", create it." << std::endl;
			auto tmp_kernel = cl::Kernel(program, entry.c_str());
			_kernels[entry] = tmp_kernel;
			return tmp_kernel;
		}
	}

private:
};

inline MyDevInfo get_device_info(size_t max_ws_in_one_group[3], cl_uint &max_compute_units)
{
	MyDevInfo dev_info;

	cl_platform_id platform;
	cl_device_id device;
	cl_uint num_devices;
	cl_int err;

	std::cerr << "== Getting device info." << std::endl;

	// 获取平台和设备
	device = get_gpu_device().get();

	// 1. 查询每个工作组的最大工作项数量
	size_t max_work_group_size;
	err = clGetDeviceInfo(
		device,
		CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(size_t),
		&max_work_group_size,
		NULL);
	if (err == CL_SUCCESS)
	{
		std::cout << "  Max work-items per group: " << max_work_group_size << std::endl;
		dev_info.device_max_group_size = max_work_group_size;
	}

	// 2. 查询每个维度的最大工作项数量
	size_t max_work_item_sizes[3];
	err = clGetDeviceInfo(
		device,
		CL_DEVICE_MAX_WORK_ITEM_SIZES,
		sizeof(max_work_item_sizes),
		max_work_item_sizes,
		NULL);
	if (err == CL_SUCCESS)
	{
		std::memcpy(max_ws_in_one_group, max_work_item_sizes, 3 * sizeof(size_t));
		std::memcpy(dev_info.device_max_work_item_size, max_work_item_sizes, 3 * sizeof(size_t));
		std::cout << "  Max work-items in each dimension: "
				  << dev_info.device_max_work_item_size[0] << " (x), "
				  << dev_info.device_max_work_item_size[1] << " (y), "
				  << dev_info.device_max_work_item_size[2] << " (z)" << std::endl;
	}

	err = clGetDeviceInfo(
		device,
		CL_DEVICE_MAX_COMPUTE_UNITS,
		sizeof(max_compute_units),
		&max_compute_units,
		NULL);
	if (err == CL_SUCCESS)
	{
		std::cout << "  Max max supported EU(compute unit) number: " << max_compute_units << std::endl;
		dev_info.device_max_compute_unit = max_compute_units;
	}

	return dev_info;
}

inline MyDevInfo get_device_info()
{
	size_t max_ws_in_one_group[3];
	cl_uint max_compute_units = 0;
	return get_device_info(max_ws_in_one_group, max_compute_units);
}