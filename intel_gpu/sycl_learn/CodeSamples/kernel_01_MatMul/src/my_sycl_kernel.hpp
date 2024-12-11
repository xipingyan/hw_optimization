#pragma once
#include <random>
#include <sycl/sycl.hpp>

#define CHECK_MALLOC(X) \
    if (nullptr == X)   \
    std::cout << "== " << __FILE__ << ":" << __LINE__ << " [Fail] can't malloc." << std::endl

struct MMParamsInput {
    using PTR = std::shared_ptr<MMParamsInput>;
    float *_a = nullptr; // input   [M*N]
    float *_b = nullptr; // Weight  [N*K]
    size_t _m = 0, _n = 0, _k = 0;

    MMParamsInput() = delete;
    MMParamsInput(size_t m, size_t n, size_t k) : _m(m), _n(n), _k(k)
    {
        // init params with random data.
        _a = (float*)malloc(sizeof(float) * m * n);
        CHECK_MALLOC(_a);
        _b = (float*) malloc(sizeof(float) * n * k);
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
        for (size_t i = 0; i < _m * _n; i++)
        {
            // std::cout << dis(gen) << ' ';
            _a[i] = dis(gen);
        }
        for (size_t i = 0; i < _n * _k; i++)
        {
            _b[i] = dis(gen);
        }
    }

    static PTR create(int m, int n, int k)
    {
        return std::make_shared<MMParamsInput>(m, n, k);
    }
};

struct MMParamsOutput
{
    using PTR = std::shared_ptr<MMParamsOutput>;
    float *_c = nullptr; // output  [M*K]
    size_t _m = 0, _k = 0;

    MMParamsOutput() = delete;
    MMParamsOutput(size_t m, size_t k) : _m(m), _k(k)
    {
        _c = (float *)malloc(sizeof(float) * m * k);
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
        for (size_t i = 0; i < _m * _k; i++)
        {
            _c[i] = 0.0f;
        }
    }

    static PTR create(int m, int k)
    {
        return std::make_shared<MMParamsOutput>(m, k);
    }
};

float matmal_kernel_ref(MMParamsInput::PTR input, MMParamsOutput::PTR output);

float matmal_kernel_1(sycl::queue &q, MMParamsInput::PTR input, MMParamsOutput::PTR output, int group_x = 16, int group_y = 16);

bool is_same(MMParamsOutput::PTR output1, MMParamsOutput::PTR output2, float T = 1e-8f);