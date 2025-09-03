#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cfloat>
#include <random>
#include <memory>
#include <omp.h>
#include <chrono>

#include "my_common.hpp"

class CGEMM_Ref
{
	int _m, _n, _k;
	float *_input = nullptr;
	float *_weight = nullptr;
	float *_output = nullptr;
	void init_random_data()
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0f, 1.0f);

		_input = new float[_m * _k];
		for (int i = 0; i < _m * _k; i++)
		{
			_input[i] = dis(gen);
		}
		_weight = new float[_k * _n];
		for (int i = 0; i < _k * _n; i++)
		{
			_weight[i] = dis(gen);
			// _weight[i] = _weight[i] / 10.0f;
		}
		_output = new float[_m * _n];
		memset(_output, 0, _m * _n * sizeof(float));
#if 0
		FILE *pf = fopen("input_mat.bin", "wb");
		for (int i = 0; i < _m * _n * _k; i++)
		{
			fwrite(&_data[i], sizeof(float), 1, pf);
		}
		fclose(pf);
#endif
	}

	CGEMM_Ref() = delete;
	void calc_output()
	{
		std::cout << "== Kernel: CPU ref: " << __FUNCTION__ << std::endl;
		std::cout << "  Input  A = " << _m << " x " << _k << std::endl;
		std::cout << "  Input  B = " << _k << " x " << _n << std::endl;
		std::cout << "  Output C = " << _m << " x " << _n << std::endl;

		auto t1 = std::chrono::high_resolution_clock::now();
		for (int m = 0; m < _m; m++)
		{
// #pragma omp parallel for num_threads(8)
			for (int n = 0; n < _n; n++)
			{
				_output[m * _n + n] = 0;
				for (int k = 0; k < _k; k++)
				{
					_output[m * _n + n] += _input[m * _k + k] * _weight[k * _n + n];
				}
			}
		}

		auto t2 = std::chrono::high_resolution_clock::now();
		auto dur = tm_diff_ms(t1, t2);
		std::cout << "  Ref time: " << dur << " ms" << std::endl;
	}

public:
	CGEMM_Ref(int m, int n, int k) : _m(m), _n(n), _k(k)
	{
		init_random_data();
		calc_output();
	}

	int get_m() { return _m; }
	int get_n() { return _n; }
	int get_k() { return _k; }

	using Ptr = std::shared_ptr<CGEMM_Ref>;
	static Ptr createPtr(int m, int n, int k)
	{
		return std::make_shared<CGEMM_Ref>(m, n, k);
	}
	~CGEMM_Ref()
	{
		if (_input)
			delete[] _input;
		if (_weight)
			delete[] _weight;
		if (_output)
			delete[] _output;
	}

	float *get_input()
	{
		return _input;
	}
	float *get_weight()
	{
		return _weight;
	}
	float *get_output()
	{
		return _output;
	}
};


inline bool is_same_buf(std::string prefix, float* output1, float*  output2, float T, bool trans_b, int M, int N, int K) {
	bool bsame = true;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			const auto &a_val = output1[i * N + j];
			const auto &b_val = trans_b ? output2[j * N + i] : output2[i * N + j];

			if (fabs(a_val - b_val) > T)
			{
				std::cout << prefix << " output1[" << i << ", " << j << "] = " << a_val << ", output2[" << i << ", " << j << "]=" << b_val << " diff = " << fabs(a_val - b_val) << std::endl;

				bsame = false;
				return false;
			}
		}
	}

	if (bsame)
		std::cout << prefix << " is same. Success." << std::endl;
	else
		std::cout << prefix << " is diff. Fail." << std::endl;
	return true;
}