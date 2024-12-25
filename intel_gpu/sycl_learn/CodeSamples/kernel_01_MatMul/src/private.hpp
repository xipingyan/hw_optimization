#pragma once
#include <algorithm>
#include <random>
#include <sycl/sycl.hpp>

#define CHECK_MALLOC(X) \
    if (nullptr == X)   \
    std::cout << "== " << __FILE__ << ":" << __LINE__ << " [Fail] can't malloc." << std::endl

template <typename IN_TYPE>
struct MMParamsInput {
    using PTR = std::shared_ptr<MMParamsInput>;
    IN_TYPE *_a = nullptr; // input   [M*N]
    IN_TYPE *_b = nullptr; // Weight  [N*K]
    size_t _m = 0, _k = 0, _n = 0;

    MMParamsInput() = delete;
    MMParamsInput(size_t m, size_t k, size_t n) : _m(m), _n(n), _k(k)
    {
        // init params with random data.
        _a = (IN_TYPE*)malloc(sizeof(IN_TYPE) * m * k);
        CHECK_MALLOC(_a);
        _b = (IN_TYPE*) malloc(sizeof(IN_TYPE) * k * n);
        CHECK_MALLOC(_b);
        init_random();
    }
    ~MMParamsInput()
    {
        if (_a)
            free(_a);
        if (_b)
            free(_b);
        _a = _b = nullptr;
    }

    void init_random()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0, 1); // uniform distribution between 0 and 1
        for (size_t i = 0; i < _m * _k; i++)
        {
            // std::cout << dis(gen) << ' ';
            _a[i] = dis(gen);
        }
        for (size_t i = 0; i < _k * _n; i++)
        {
            _b[i] = dis(gen);
        }
    }

    static PTR create(int m, int k, int n)
    {
        return std::make_shared<MMParamsInput>(m, k, n);
    }

    void print()
    {
        auto print_matrix = [](IN_TYPE *data, size_t m, size_t n)
        {
            for (size_t i = 0; i < m; i++)
            {
                if (i > 6u)
                {
                    std::cout << "  ,\n  ,\n";
                    break;
                }
                std::cout << "  ";
                for (size_t j = 0; j < n; j++)
                {
                    if (j > 6u)
                    {
                        std::cout << ", , ,";
                        break;
                    }
                    // static_cast<sycl::half>(accFloat[i]);.
                    std::cout << static_cast<float>(data[i * n + j]) << ", ";
                }
                std::cout << std::endl;
            }
        };
        
        std::cout << " == input A:" << std::endl;
        print_matrix(_a, _m, _k);
        std::cout << " == input B:" << std::endl;
        print_matrix(_b, _k, _n);
    }
};

template <typename OUT_TYPE>
struct MMParamsOutput
{
    using PTR = std::shared_ptr<MMParamsOutput>;
    OUT_TYPE *_c = nullptr; // output  [M*K]
    size_t _m = 0, _n = 0;

    MMParamsOutput() = delete;
    MMParamsOutput(size_t m, size_t n) : _m(m), _n(n)
    {
        _c = (OUT_TYPE *)malloc(sizeof(OUT_TYPE) * m * n);
        CHECK_MALLOC(_c);
        init_random();
    }
    ~MMParamsOutput()
    {
        if (_c)
            free(_c);
        _c = nullptr;
    }

    void init_random()
    {
        for (size_t i = 0; i < _m * _n; i++)
        {
            _c[i] = 0.0f;
        }
    }

    static PTR create(int m, int n)
    {
        return std::make_shared<MMParamsOutput>(m, n);
    }
};

// Reference implementation.
float matmal_kernel_ref(MMParamsInput<float>::PTR input, MMParamsOutput<float>::PTR output);
float matmal_kernel_openblas(MMParamsInput<float>::PTR input, MMParamsOutput<float>::PTR output);

float matmal_kernel_1(sycl::queue &q, MMParamsInput<float>::PTR input, MMParamsOutput<float>::PTR output, int group_x = 16, int group_y = 16);
float matmal_kernel_1_inp_f16(sycl::queue &q, MMParamsInput<sycl::half>::PTR input, MMParamsOutput<float>::PTR output, int group_x = 16, int group_y = 16);

float add_kernel_1(sycl::queue &q, float *data, size_t len, float &output, int group_x = 1);
float add_kernel_1_f16(sycl::queue &q, sycl::half *data, size_t len, sycl::half &output, int group_x);

bool is_same(std::string prefix, MMParamsOutput<float>::PTR output1, MMParamsOutput<float>::PTR output2, float T = 1e-8f, bool trans_b = false);

MMParamsInput<sycl::half>::PTR cvt_f32_to_half(MMParamsInput<float>::PTR ptr);