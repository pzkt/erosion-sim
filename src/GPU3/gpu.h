#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "helper.h"

class Gpu3
{
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize);
    static void generateHeightmap(float *heightmap, int size, PerlinParams p);

    static void shutdown();

    static void CHECK_CUDA(cudaError_t call, const char *msg = "")
    {
        if (call != cudaSuccess)
        {
            std::cerr << msg << "\nCUDA error at " << __LINE__ << ": " << cudaGetErrorString(call) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

private:
    // ---------- Persistent device buffers ----------
    static float *d_map;
    static size_t d_mapBytes;

    static int *d_brushIndices;
    static float *d_brushWeights;
    static int brushLength;
    static int brushMapSize;

    static curandState *d_rngStates;
    static int rngStateCount;

    // ---------- Pinned host buffer ----------
    static float *h_pinnedMap;
    static size_t h_pinnedBytes;

    // ---------- Stream ----------
    static cudaStream_t stream;

    // ---------- Internal helpers ----------
    static void ensureStream();
    static void ensurePinnedHost(size_t bytes);
    static void ensureDeviceMap(size_t bytes);
    static void ensureRNG(int totalThreads);
    static void uploadBrush(const ErosionParams &p, int mapSize);
};