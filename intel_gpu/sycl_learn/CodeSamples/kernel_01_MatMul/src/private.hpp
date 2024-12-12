#pragma once
#include <algorithm>
#include <random>
#include <sycl/sycl.hpp>

#define CHECK_MALLOC(X) \
    if (nullptr == X)   \
    std::cout << "== " << __FILE__ << ":" << __LINE__ << " [Fail] can't malloc." << std::endl

struct MMParamsInput {
    using PTR = std::shared_ptr<MMParamsInput>;
    float *_a = nullptr; // input   [M*N]
    float *_b = nullptr; // Weight  [N*K]
    size_t _m = 0, _k = 0, _n = 0;

    MMParamsInput() = delete;
    MMParamsInput(size_t m, size_t k, size_t n) : _m(m), _n(n), _k(k)
    {
        // init params with random data.
        _a = (float*)malloc(sizeof(float) * m * k);
        CHECK_MALLOC(_a);
        _b = (float*) malloc(sizeof(float) * k * n);
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
        std::cout << " == input a:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)3, _m); i++)
        {
            std::cout << "  ";
            for (size_t j = 0; j < std::min((size_t)3u, _k); j++)
            {
                std::cout << _a[i * _k + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << " == input b:" << std::endl;
        for (size_t i = 0; i < std::min((size_t)3u, _k); i++)
        {
            std::cout << "  ";
            for (size_t j = 0; j < std::min((size_t)3u, _n); j++)
            {
                std::cout << _b[i * _n + j] << ", ";
            }
            std::cout << std::endl;
        }
    }
};

struct MMParamsOutput
{
    using PTR = std::shared_ptr<MMParamsOutput>;
    float *_c = nullptr; // output  [M*K]
    size_t _m = 0, _n = 0;

    MMParamsOutput() = delete;
    MMParamsOutput(size_t m, size_t n) : _m(m), _n(n)
    {
        _c = (float *)malloc(sizeof(float) * m * n);
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
float matmal_kernel_ref(MMParamsInput::PTR input, MMParamsOutput::PTR output);
float matmal_kernel_openblas(MMParamsInput::PTR input, MMParamsOutput::PTR output);

float matmal_kernel_1(sycl::queue &q, MMParamsInput::PTR input, MMParamsOutput::PTR output, int group_x = 16, int group_y = 16);

float add_kernel_1(sycl::queue &q, float *data, size_t len, float &output, int group_x = 1);
float add_kernel_1_f16(sycl::queue &q, sycl::half *data, size_t len, sycl::half &output, int group_x);

bool is_same(std::string prefix, MMParamsOutput::PTR output1, MMParamsOutput::PTR output2, float T = 1e-8f, bool trans_b = false);