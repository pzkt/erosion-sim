#include "gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <helper.h>
#include <random>

__device__ inline float fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ inline float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

__device__ inline float grad(int hash, float x, float y)
{
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

__device__ inline int hash2D(int x, int y)
{
    int h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return h;
}

__device__ float perlin(float x, float y)
{
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);

    float fx = x - x0;
    float fy = y - y0;

    float u = fade(fx);
    float v = fade(fy);

    float n00 = grad(hash2D(x0, y0), fx, fy);
    float n10 = grad(hash2D(x0 + 1, y0), fx - 1, fy);
    float n01 = grad(hash2D(x0, y0 + 1), fx, fy - 1);
    float n11 = grad(hash2D(x0 + 1, y0 + 1), fx - 1, fy - 1);

    float nx0 = lerp(n00, n10, u);
    float nx1 = lerp(n01, n11, u);

    return lerp(nx0, nx1, v);
}

__device__ float fbm(float x, float y, const PerlinParams &p, const float2 *offsets)
{
    float frequency = 1.0f / p.initialScale;
    float amplitude = 1.0f;
    float value = 0.0f;
    float ampSum = 0.0f;

    for (int o = 0; o < p.numOctaves; ++o)
    {
        float nx = offsets[o].x + x * frequency;
        float ny = offsets[o].y + y * frequency;

        float n = perlin(nx, ny);

        value += n * amplitude;
        ampSum += amplitude;

        amplitude *= p.persistence;
        frequency *= p.lacunarity;
    }

    return value / ampSum;
}

__device__ inline float ridged(float n)
{
    n = fabsf(n);
    n = 1.0f - n;
    return n * n;
}

__device__ float ridgedFBM(float x, float y, const PerlinParams &p, const float2 *offsets)
{
    float frequency = 1.0f / p.initialScale;
    float amplitude = 0.5f;
    float value = 0.0f;
    float weight = 1.0f;

    for (int o = 0; o < p.numOctaves; ++o)
    {
        float nx = offsets[o].x + x * frequency;
        float ny = offsets[o].y + y * frequency;

        float n = perlin(nx, ny);
        float r = ridged(n);

        r *= weight;
        weight = fminf(fmaxf(r * 2.0f, 0.0f), 1.0f);

        value += r * amplitude;

        frequency *= p.lacunarity;
        amplitude *= p.persistence;
    }

    return value;
}

__device__ inline float erosionBias(float h)
{
    h = (h + 1.0f) * 0.5f;
    return powf(h, 1.4f);
}

__global__ void generateKernel(float *d_map, int size, PerlinParams p, const float2 *offsets)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size)
        return;

    float base = fbm((float)x, (float)y, p, offsets);
    float ridge = ridgedFBM((float)x + 1000.0f, (float)y + 1000.0f, p, offsets);

    float h = base * 0.5f + ridge * 0.5f;
    h = erosionBias(h);

    d_map[y * size + x] = h;
}

void Gpu::generateHeightmap(float *heightmap, int size, PerlinParams p)
{
    const size_t mapBytes = size * size * sizeof(float);
    const size_t offsetsBytes = p.numOctaves * sizeof(float2);

    float *d_map = nullptr;
    float2 *d_offsets = nullptr;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_map, mapBytes));
    CHECK_CUDA(cudaMalloc(&d_offsets, offsetsBytes));

    std::mt19937 rgen(70);
    std::uniform_real_distribution<float> dist(-10000.f, 10000.f);

    std::vector<Vector2> offsets(p.numOctaves);
    for (auto &v : offsets)
        v = {dist(rgen), dist(rgen)};

    CHECK_CUDA(cudaMemcpy(d_offsets, offsets.data(), offsetsBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);
    generateKernel<<<grid, block>>>(d_map, size, p, d_offsets);

    CHECK_CUDA(cudaGetLastError(), "generateKernel launch failed");
    CHECK_CUDA(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    CHECK_CUDA(cudaMemcpy(heightmap, d_map, mapBytes, cudaMemcpyDeviceToHost), "cudaMemcpy D->H failed");
    cudaFree(d_map);
}
