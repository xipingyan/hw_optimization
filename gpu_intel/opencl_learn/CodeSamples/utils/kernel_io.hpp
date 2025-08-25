#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline std::string load_kernel_source_codes(std::string kernel_source_fn)
{
    std::ifstream file(kernel_source_fn, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << kernel_source_fn << std::endl;
        return "";
    }

    std::string source;
    file.seekg(0, std::ios::end);
    source.reserve(file.tellg());
    file.seekg(0, std::ios::beg);

    source.assign((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());

    file.close();
    return source;
}

// inline std::string get_kernel_path(std::string kernel_fn) {
//     std::string path = "" + kernel_fn;
// }

inline void dump_kernel_bin(cl::Program &program, std::string out_fn = "ocl_kernel_vec_add.bin")
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