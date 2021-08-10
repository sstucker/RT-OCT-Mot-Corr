#pragma once
#include <vector>
#include <complex>

inline int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

inline std::complex<float> fftwf_conj(std::complex<float> imag)
{
    return std::conj(imag);
}

template<typename T>
inline std::vector<float> linspace(T start_in, T end_in, int num_in)
{
    std::vector<float> linspaced;
    float start = static_cast<float>(start_in);
    float end = static_cast<float>(end_in);
    float num = static_cast<float>(num_in);
    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }
    float delta = (end - start) / (num - 1);
    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}