#pragma once

#include <CL/opencl.hpp>
#include <iostream>

#if 1
#define DEBUG_LOG std::cout
#else
#define DEBUG_LOG std::cout << __FUNCTION__ << ":" << __LINE__ << " log: "
#endif

inline void print_nd_range(cl::NDRange gs, std::string prefix = std::string()) {
    std::cout << prefix.c_str() << " " << gs[0] << ", " << gs[1] << ", " << gs[2] << std::endl;
}