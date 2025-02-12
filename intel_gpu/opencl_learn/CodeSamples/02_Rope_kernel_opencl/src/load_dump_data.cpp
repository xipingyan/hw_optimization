#include <fstream>
#include <iostream>
#include <string>
#include <bits/stdc++.h>

#include "private.hpp"

std::string DumpData::to_string()
{
    std::string ret;
    ret = "shape[";
    for (auto s : shape)
    {
        ret += std::to_string(s) + ", ";
    }
    ret += "], format=" + format + ", data size=" + std::to_string(data.size()) + ", data=";
    for (auto i = 0; i < std::min((size_t)2u, data.size()); i++)
    {
        ret += std::to_string(data[i]) + ", ";
    }
    return ret;
}

ushort *DumpData::to_half()
{
    auto buf = (ushort *)malloc(data.size() * sizeof(ushort));
    for (size_t i = 0; i < data.size(); i++)
    {
        buf[i] = float_to_half(data[i]);
    }
    return buf;
}

int* DumpData::to_int()
{
    auto buf = (int *)malloc(data.size() * sizeof(int));
    for (size_t i = 0; i < data.size(); i++)
    {
        buf[i] = static_cast<int>(data[i]);
    }
    return buf;
}

std::vector<std::string> str_split(std::string str, char delimiter)
{
    std::vector<std::string> strs;
    // Create a stringstream object to str
    std::stringstream ss(str);

    // Temporary object to store the splitted string
    std::string t;

    // Splitting the str string by delimiter
    while (getline(ss, t, delimiter))
        strs.push_back(t);
    return strs;
}

DumpData load_dump_data(std::string fn) {
    DumpData dd;
    std::ifstream input_f(fn);
    std::string line;
    // First line
    if (getline(input_f, line)) {
        // example str: shape: [b:1, f:6, x:64, y:14, z:1, w:1, u:1, v:1, g:1] (count: 5376, original format: bfyx) raw data
        // Parse shape.
        auto p1 = line.find('[');
        auto p2 = line.find(']');
        auto shape_str = line.substr(p1 + 1, p2 - p1 - 1);
        auto strs = str_split(shape_str, ',');
        for (auto ss : strs) {
            auto vals = str_split(ss, ':');
            if (vals.size() != 2) {
                std::cout << "Parse error: vals.size()=" << vals.size() << ", but expected is 2." << std::endl;
                exit(0);
            }

            dd.shape.push_back(std::atoi(vals[1].c_str()));
        }
        // Parse format
        p1 = line.find("format: ");
        p2 = line.find(')');
        dd.format = line.substr(p1 + 8, p2 - p1 - 8);
    }

    while (getline(input_f, line))
    {
        dd.data.emplace_back(atof(line.c_str()));
    }
    input_f.close();
    
    std::cout << "Parsed dump data=" << dd.to_string() << std::endl;
    return dd;
}