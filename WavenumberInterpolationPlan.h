#pragma once
#include "fftw3.h"
#include <vector>
#include <algorithm>
#include "Utils.h"
#include <Windows.h>

class WavenumberInterpolationPlan
{

public:
    int aline_size;
    double intpDk;
	std::vector<std::vector<int>> interp_map;  // Map of nearest neighbors
	std::vector<float> linear_in_lambda;  // Linear wavelength space
	std::vector<float> linear_in_k;  // Linear wavenumber space points to interpolate
	float d_lam;

    WavenumberInterpolationPlan() {}

    WavenumberInterpolationPlan(int aline_size, double intpDk)
    {
        linear_in_lambda = linspace(1 - (intpDk / 2), 1 + (intpDk / 2), aline_size);
        for (int i = 0; i < aline_size; i++)
        {
            linear_in_lambda[i] = 1 / linear_in_lambda[i];
        }
        float min_lam = *std::min_element(linear_in_lambda.begin(), linear_in_lambda.end());
        float max_lam = *std::max_element(linear_in_lambda.begin(), linear_in_lambda.end());
        linear_in_k = linspace(min_lam, max_lam, aline_size);

        d_lam = linear_in_lambda[1] - linear_in_lambda[0];

        interp_map = std::vector<std::vector<int>>(2);
        interp_map[0] = std::vector<int>(aline_size);  // Left nearest-neighbor
        interp_map[1] = std::vector<int>(aline_size);  // Right nearest-neighbor

        //(Naively, but only once) find nearest upper and lower indices for linear interpolation
        for (int i = 0; i < aline_size; i++)  // For each k-linearized interpolation point
        {
            // Calculate distance vector
            std::vector<float> distances = std::vector<float>(aline_size);
            for (int j = 0; j < aline_size; j++)  // For each linear-in-wavelength point
            {
                distances[j] = std::abs(linear_in_lambda[j] - linear_in_k[i]);
            }
            auto nn = std::min_element(distances.begin(), distances.end()) - distances.begin();  // Find closest point
            if (nn == 0)
            {
                interp_map[0][i] = 0;
                interp_map[1][i] = 0;
            }
            else if (nn == aline_size - 1)
            {
                interp_map[0][i] = aline_size - 1;
                interp_map[1][i] = aline_size - 1;
            }
            else if (linear_in_lambda[nn] >= linear_in_k[nn])
            {
                interp_map[0][i] = nn - 1;
                interp_map[1][i] = nn;
            }
            else if (linear_in_lambda[nn] < linear_in_k[nn])
            {
                interp_map[0][i] = nn;
                interp_map[1][i] = nn + 1;
            }
        }
    }

};