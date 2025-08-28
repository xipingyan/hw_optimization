#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cfloat>

class CMatDiagMax
{
	int _m, _n;
	float* _data;

public:
	CMatDiagMax() = delete;
	CMatDiagMax(int m, int n, float *data) : _m(m), _n(n), _data(data)
	{
		assert(_m == _n);
	}
	void get_max_val(float &max_val)
	{
		max_val = -FLT_MAX;
		for (int i = 0; i < _m; i++)
		{
			auto id = i * _m + i;
			max_val = std::max(max_val, _data[id]);
		}
	}
};