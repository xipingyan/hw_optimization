
#include "private.hpp"
#include <fstream>
#include <thread>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <sycl/ext/oneapi/backend/level_zero.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

inline std::string usm_alloc_2_str(sycl::usm::alloc mem_type)
{
	switch (mem_type)
	{
	case sycl::usm::alloc::device:
		return "sycl::usm::alloc::device";
	case sycl::usm::alloc::host:
		return "sycl::usm::alloc::host";
	case sycl::usm::alloc::shared:
		return "sycl::usm::alloc::shared";
	case sycl::usm::alloc::unknown:
		return "sycl::usm::alloc::unknown";
	}
	return std::string();
}

class sycl_args
{
public:
    sycl_args() = delete;
    sycl_args(sycl::buffer<uint8_t, 1, sycl::image_allocator, void> buf, bool isOutput) : _isBuf(true), _buf(buf), _val(0), _isOutput(isOutput)
    {
    }
    sycl_args(int val) : _isBuf(false), _buf(0, 1), _val(val)
    {
    }
    bool _isBuf;
    sycl::buffer<uint8_t, 1, sycl::image_allocator, void> _buf;
    int _val = 0; // if isBuf == false;
    bool _isOutput = false;
    friend std::ostream &operator<<(std::ostream &os, const sycl_args &bf);
};

std::ostream &operator<<(std::ostream &os, const sycl_args &bf)
{
    os << "sycl_args(_isBuf = " << bf._isBuf << ", _val = " << bf._val << ", _isOutput = " << bf._isOutput << ")";
    return os;
};

void my_set_args(sycl::handler &cgh, size_t idx, sycl_args buf) {
    if (buf._isOutput)
    {
        // Last one is output.
        sycl::accessor acc_param{buf._buf, cgh, sycl::read_write};
        cgh.set_arg(idx, acc_param);
    }
    else
    {
        if (buf._isBuf)
        {
            sycl::accessor acc_param{buf._buf, cgh, sycl::read_only};
            cgh.set_arg(idx, acc_param);
        }
        else
        {
            cgh.set_arg(idx, buf._val);
        }
    }
}

// Launch OpenCL, online compile to Sycl interface.
// Refer: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler_opencl.asciidoc
sycl::event launchOpenCLKernelOnline(sycl::queue &q, std::vector<std::pair<sycl::buffer<uint8_t, 1>, bool>>& params, std::string source, sycl::event &dep_event)
{
    std::cout << "== Start to test launch OpenCL kernel and compile online." << std::endl;

    std::cout << "  == Start to kernel_bundle opencl source" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source> kb_src =
        syclex::create_kernel_bundle_from_source(
            q.get_context(),
            syclex::source_language::opencl,
            source);

    // Compile and link the kernel from the source definition.
    std::cout << "  == Start to kernel_bundle kb_src" << std::endl;
    sycl::kernel_bundle<sycl::bundle_state::executable> kb_exe =
        syclex::build(kb_src);

    // Get a "kernel" object representing the kernel defined in the
    // source string.
    std::cout << "  == Start to get sycl::kernel" << std::endl;
    sycl::kernel k = kb_exe.ext_oneapi_get_kernel("rope_ref_11982042700243959200_0_0__sa");

    std::cout << "  == Start to submit" << std::endl;
    return q.submit([&](sycl::handler &cgh)
                    {
                        cgh.depends_on(dep_event);
                        for (int i = 0; i < params.size(); i++)
                        {
                            sycl::accessor acc_param{params[i].first, cgh, sycl::read_write};
                            cgh.set_arg(i, acc_param);
                            // my_set_args(cgh, i, inputs_buf[i]);
                        }

                        // Invoke the kernel over an nd-range.
                        sycl::nd_range ndr = sycl::nd_range{sycl::range{192, 14, 1}, sycl::range{192, 2, 1}};
                        // sycl::nd_range ndr = sycl::nd_range{sycl::range{1, 14, 192}, sycl::range{1, 2, 192}};//opencl backend.
                        cgh.parallel_for(ndr, k);
                    });
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
	return std::string();
}

int test_02_main()
{
    std::cout << "Start to test call SPIR-V kernel(converted from opencl kernel)." << std::endl;

    // char env[] = "ONEAPI_DEVICE_SELECTOR=OPENCL:GPU";
    // putenv(env);

    auto queue = sycl::queue(sycl::gpu_selector_v);
    std::cout << "== Using "
              << queue.get_device().get_info<sycl::info::device::name>()
              << ", Backend: " << queue.get_backend()
              << std::endl;
    // auto order_properity = queue.get_property<sycl::property::queue::in_order>();
    // std::cout << "  == order_properity = " << order_properity.getKind() << std::endl;

    // input param:
    std::string kernel_path = "../02_sycl_ocl_interoperate/src/kernel_rope_ref/";
	std::string kernel_source = load_kernel(kernel_path + "SYCL_LZ_program_1_bucket_0_part_53_8491392767821923070.cl");

    DumpData input0;
	input0.data = std::vector<float>({1, 6, 1, 1, 1, 1, 14, 64, 0, 4, 1, 1, 1, 1, 1, 1, 6, 64, 1, 1, 1, 1, 1, 1, 6, 64, 1, 14, 1, 1, 1, 1, 6, 64});
	input0.format = "bfyx";
	input0.shape = {34};

	auto input1 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src0.txt");
	auto input2 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src1.txt");
	auto input3 = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_src2.txt");
	auto output_expected = load_dump_data(kernel_path + "program1_network1_0_rope___module.model.layers.0.self_attn_aten__add_Add_dst0.txt");

	auto buf0 = input0.to_int(queue);
	auto buf1 = input1.to_half(queue);
	auto buf2 = input2.to_half(queue);
	auto buf3 = input3.to_half(queue);
	// for (size_t i = 0; i < input2.data.size(); i++) {
	// 	std::cout << buf2[i] << ", ";
	// }
	// std::cout << std::endl;
	auto *output_buf = (sycl::half *)sycl::malloc_shared(output_expected.data.size() * sizeof(sycl::half), queue.get_device(), queue.get_context());
    auto *output_buf_host = (sycl::half *)sycl::malloc_host(output_expected.data.size() * sizeof(sycl::half), queue.get_context());

    sycl::buffer param_0(reinterpret_cast<uint8_t *>(buf0), sycl::range{input0.data.size() * sizeof(int)});
    sycl::buffer param_1(reinterpret_cast<uint8_t *>(buf1), sycl::range{input1.data.size() * sizeof(sycl::half)});
    sycl::buffer param_2(reinterpret_cast<uint8_t *>(buf2), sycl::range{input2.data.size() * sizeof(sycl::half)});
    sycl::buffer param_3(reinterpret_cast<uint8_t *>(buf3), sycl::range{input3.data.size() * sizeof(sycl::half)});
    std::vector<std::pair<sycl::buffer<uint8_t, 1>, bool>> params;
    params.push_back({param_0, false});
    params.push_back({param_1, false});
    params.push_back({param_2, false});
    params.push_back({param_3, false});
    sycl::buffer param_4((uint8_t *)(output_buf), sycl::range{output_expected.data.size() * sizeof(sycl::half)});
    params.push_back({param_4, true});

    sycl::event event1;
    // 2: OpenCL online kernel: Z = 2 * Z + X;

    output_buf[0] = 20;
  
    std::cout << "== launchOpenCLKernelOnline." << std::endl;
    auto event2 = launchOpenCLKernelOnline(queue, params, kernel_source, event1);
    event2.wait();

    std::cout << "== sycl copy kernel." << std::endl;
    queue.submit([&](sycl::handler &cgh){
        cgh.depends_on(event2);
        cgh.parallel_for(sycl::range<1>(output_expected.data.size()), [=](sycl::id<1> i)
                         { output_buf_host[i] = output_buf[i]; });
    }).wait();
    // queue.parallel_for(sycl::range<1>(output_expected.data.size()), [=](sycl::id<1> i) {
    //     output_buf_host[i] = output_buf[i];
    // }).wait();

    std::cout << "== Start to compare result." << std::endl;
    bool is_expected = true;
    for (size_t i = 0; i < 5; i++)
    {
        if (abs(output_expected.data[i] - output_buf_host[i]) > 0)
        {
            std::cout << "== Result [" << i << "] diff: " << abs(output_expected.data[i] - output_buf_host[i]) << ", expect: " << output_expected.data[i] << ", result=" << output_buf_host[i] << std::endl;
            is_expected = false;
        }
    }

    std::cout << (is_expected ? "Success!\n" : "Fail!\n");
    return 0;
}

int main()
{
    std::cout << "== Debug MACRO tip:" << std::endl;
    std::cout << "  Default:        test accuracy only." << std::endl;
    std::cout << "  PERFORMANCE=1:  Test performance only." << std::endl;

    // test_sycl_olc_interoperate_l0_backend();
    // test_sycl_olc_interoperate_l0_backend_reoder_weights();
    // test_sycl_olc_interoperate_ocl_backend();
    test_sycl_olc_interoperate_l0_backend_rope_ref();
    // test_02_main();
    return 0;
}