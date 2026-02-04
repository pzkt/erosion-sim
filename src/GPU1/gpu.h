#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "helper.h"

class Gpu1
{
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize);
    static void generateHeightmap(float *heightmap, int size, PerlinParams p);

    static void CHECK_CUDA(cudaError_t call, const char *msg = "")
    {
        if (call != cudaSuccess)
        {
            std::cerr << msg << "\nCUDA error at " << __LINE__ << ": " << cudaGetErrorString(call) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};