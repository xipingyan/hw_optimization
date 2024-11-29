#pragma once

#include <CL/sycl.hpp>

#include <my_log.hpp>

inline void print_device_info(sycl::queue queue) {
    DEBUG_LOG << "Device info:" << std::endl;

#define PRINT_ITM(ITM) std::cout << "  == " << #ITM << " : " << queue.get_device().get_info<sycl::info::device::ITM>() << std::endl
    PRINT_ITM(name);
    PRINT_ITM(vendor);
    PRINT_ITM(vendor_id);
    PRINT_ITM(driver_version);
    PRINT_ITM(version);

    PRINT_ITM(max_compute_units);
    PRINT_ITM(max_work_item_dimensions);
    PRINT_ITM(max_work_group_size);
    PRINT_ITM(max_num_sub_groups);

    PRINT_ITM(max_clock_frequency);
    PRINT_ITM(max_mem_alloc_size);
    PRINT_ITM(max_samplers);
    PRINT_ITM(max_parameter_size);
    PRINT_ITM(global_mem_cache_line_size);
    PRINT_ITM(global_mem_cache_size);
    PRINT_ITM(global_mem_size);
    PRINT_ITM(local_mem_size);

    std::cout << std::endl;
}

inline void sycl_ls()
{
    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }
}