#include "gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <helper.h>
#include <random>

__device__ __forceinline__ float fade(float t)
{
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

__device__ __forceinline__ float fadeDeriv(float t)
{
    return 30.f * t * t * (t * (t - 2.f) + 1.f);
}

__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

__device__ __forceinline__ void grad2(
    int hash, float &gx, float &gy)
{
    switch (hash & 7)
    {
    case 0:
        gx = 1;
        gy = 1;
        break;
    case 1:
        gx = -1;
        gy = 1;
        break;
    case 2:
        gx = 1;
        gy = -1;
        break;
    case 3:
        gx = -1;
        gy = -1;
        break;
    case 4:
        gx = 1;
        gy = 0;
        break;
    case 5:
        gx = -1;
        gy = 0;
        break;
    case 6:
        gx = 0;
        gy = 1;
        break;
    default:
        gx = 0;
        gy = -1;
        break;
    }
}

__device__ __forceinline__ int hash2(int x, int y)
{
    int h = x * 374761393 + y * 668265263;
    h = (h ^ (h >> 13)) * 1274126177;
    return h;
}

__device__ __forceinline__ void rotateAndScale(float &x, float &y)
{
    // mat2(0.8, -0.6,
    //      0.6,  0.8)
    float nx = 0.8f * x - 0.6f * y;
    float ny = 0.6f * x + 0.8f * y;
    x = nx * 2.0f;
    y = ny * 2.0f;
}

__device__ float perlinNoiseGrad(
    float x, float y,
    const PerlinParams &,
    float &dx, float &dy)
{
    // Integer lattice points
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Local coordinates
    float fx = x - x0;
    float fy = y - y0;

    float u = fade(fx);
    float v = fade(fy);
    float du = fadeDeriv(fx);
    float dv = fadeDeriv(fy);

    // Gradients at corners
    float g00x, g00y, g10x, g10y, g01x, g01y, g11x, g11y;
    grad2(hash2(x0, y0), g00x, g00y);
    grad2(hash2(x1, y0), g10x, g10y);
    grad2(hash2(x0, y1), g01x, g01y);
    grad2(hash2(x1, y1), g11x, g11y);

    // Distance vectors
    float dx0 = fx, dy0 = fy;
    float dx1 = fx - 1.f, dy1 = fy;
    float dx2 = fx, dy2 = fy - 1.f;
    float dx3 = fx - 1.f, dy3 = fy - 1.f;

    // Dot products
    float n00 = g00x * dx0 + g00y * dy0;
    float n10 = g10x * dx1 + g10y * dy1;
    float n01 = g01x * dx2 + g01y * dy2;
    float n11 = g11x * dx3 + g11y * dy3;

    // Interpolate noise value
    float nx0 = lerp(n00, n10, u);
    float nx1 = lerp(n01, n11, u);
    float nxy = lerp(nx0, nx1, v);

    // d/dx
    float dnx0_dx = g00x + u * (g10x - g00x) + du * (n10 - n00);
    float dnx1_dx = g01x + u * (g11x - g01x) + du * (n11 - n01);
    dx = dnx0_dx + v * (dnx1_dx - dnx0_dx);

    // d/dy
    float dnx0_dy = g00y + u * (g10y - g00y);
    float dnx1_dy = g01y + u * (g11y - g01y);
    dy = dnx0_dy + v * (dnx1_dy - dnx0_dy) + dv * (nx1 - nx0);

    return nxy;
}

static __device__ float fbm(float x, float y, const PerlinParams &p)
{
    float amplitude = 1.0f;
    float height = 0.0f;

    // domain warp accumulator (d in GLSL)
    float dx_acc = 0.0f;
    float dy_acc = 0.0f;

    // initial scale
    x /= p.initialScale;
    y /= p.initialScale;

    for (int i = 0; i < p.numOctaves; ++i)
    {
        float dx, dy;

        // n.x = noise value
        // n.yz = noise gradient
        float n = perlinNoiseGrad(x, y, p, dx, dy);

        // accumulate gradient (domain warping)
        dx_acc += dx;
        dy_acc += dy;

        float warpAttenuation = 1.0f + (dx_acc * dx_acc + dy_acc * dy_acc);

        height += amplitude * n / warpAttenuation;

        amplitude *= p.persistence;

        // rotate + increase frequency
        rotateAndScale(x, y);
    }

    return height;
}

static __global__ void generateKernel(float *d_map, int size, PerlinParams p, const float2 *offsets)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size || y >= size)
        return;

    float h = fbm((float)x, (float)y, p) * 0.6 + 0.6;

    d_map[y * size + x] = h;
}

void Gpu1::generateHeightmap(float *heightmap, int size, PerlinParams p)
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
    cudaFree(d_offsets);
}
