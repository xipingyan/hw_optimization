// #include <cuda_runtime_api.h>
// #include <cuda_device_runtime_api.h>
#include <iostream>

void list_one_gpu_info(int devID) {
	// cudaError_t rt = cudaSetDevice(devID);
	// if (cudaSuccess != rt) {
	// 	std::cout << "  cudaSetDevice(" << devID << ") fail, return " << rt << std::endl;
	// 	return;
	// }

	// // Get the ID of the currently selected active CUDA device
	// int device = devID;
	// cudaGetDevice(&device);

	// // Fetch its properties
	// cudaDeviceProp props;
	// cudaGetDeviceProperties(&props, device);

	// /* 
	// We only print the most fundamental properties here. cudaDeviceProp 
	// contains a long range of indicators to check for different things
	// that your GPU may or may not support, as well as factors for 
	// performance. However, the most essential property to know about is
	// the compute capability of the device. 
	// */
	// std::cout << "  Model: " << props.name << std::endl;
	// std::cout << "  Compute capability: " << props.major << "." << props.minor << std::endl;
	// std::cout << "  Memory: " << props.totalGlobalMem / float(1 << 30) << " GiB" << std::endl;
	// std::cout << "  Multiprocessors: " << props.multiProcessorCount << std::endl;
	// std::cout << "  Clock rate: " << props.clockRate / float(1'000'000) << " GHz" << std::endl;
}

/*
Before you use your GPU to do work, you should know the 
most essential things about its capabilities.
*/
int main()
{
	std::cout << "Hello levelzero" << std::endl;
	// // Count CUDA-capable devices on the system
	// int numDevices;
	// cudaGetDeviceCount(&numDevices);

	// if (numDevices == 0)
	// {
	// 	std::cout << "You have no CUDA devices available!" << std::endl;
	// 	return -1;
	// }

	// for (auto n = 0; n < numDevices; n++) {
	// 	std::cout << "== GPU:" << n << std::endl;
	// 	list_one_gpu_info(n);
	// }
	return 0;
}

/*
Exercises:
1) Change the behavior such that the properties are not just printed for one, but all available CUDA devices you have!
(Even if you have just one)
2) Print a few more interesting properties and read up in the specification what they mean.
*/
