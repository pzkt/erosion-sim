#include "gpu.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

// Grid/flux-based hydraulic erosion implementation.
// This implementation is data-parallel and operates per-cell using a 4-way flux stencil.

static inline __device__ int cellId(int x, int y, int size) { return y * size + x; }

static void generateBrush(int brushRadius, std::vector<int> &brushIndexOffsets, std::vector<float> &brushWeights, int mapSize)
{
    int r = brushRadius;
    float weightSum = 0.0f;

    for (int by = -r; by <= r; by++)
    {
        for (int bx = -r; bx <= r; bx++)
        {
            float sqr = float(bx * bx + by * by);
            if (sqr <= r * r)
            {
                brushIndexOffsets.push_back(by * mapSize + bx);
                float w = 1.0f - std::sqrt(sqr) / float(r);
                weightSum += w;
                brushWeights.push_back(w);
            }
        }
    }
    // normalize weights
    for (float &w : brushWeights)
        w /= weightSum;
}

// Compute per-cell outflows to neighbors based on height+water differences.
static __global__ void computeOutFluxes(const float *height, const float *water, float *fluxN, float *fluxS, float *fluxE, float *fluxW, int mapSize, float flowFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = mapSize * mapSize;
    if (idx >= n)
        return;

    int x = idx % mapSize;
    int y = idx / mapSize;

    float h = height[idx] + water[idx];
    // neighbours
    // North (y-1)
    if (y > 0)
    {
        float hn = height[cellId(x, y - 1, mapSize)] + water[cellId(x, y - 1, mapSize)];
        float dh = h - hn;
        fluxN[idx] = fmaxf(0.0f, dh * flowFactor);
    }
    else
        fluxN[idx] = 0.0f;

    // South (y+1)
    if (y < mapSize - 1)
    {
        float hs = height[cellId(x, y + 1, mapSize)] + water[cellId(x, y + 1, mapSize)];
        float dh = h - hs;
        fluxS[idx] = fmaxf(0.0f, dh * flowFactor);
    }
    else
        fluxS[idx] = 0.0f;

    // East (x+1)
    if (x < mapSize - 1)
    {
        float he = height[cellId(x + 1, y, mapSize)] + water[cellId(x + 1, y, mapSize)];
        float dh = h - he;
        fluxE[idx] = fmaxf(0.0f, dh * flowFactor);
    }
    else
        fluxE[idx] = 0.0f;

    // West (x-1)
    if (x > 0)
    {
        float hw = height[cellId(x - 1, y, mapSize)] + water[cellId(x - 1, y, mapSize)];
        float dh = h - hw;
        fluxW[idx] = fmaxf(0.0f, dh * flowFactor);
    }
    else
        fluxW[idx] = 0.0f;
}

// Apply fluxes: compute net water change for each cell from outgoing and incoming fluxes.
static __global__ void applyFluxes(float *water, const float *fluxN, const float *fluxS, const float *fluxE, const float *fluxW, int mapSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = mapSize * mapSize;
    if (idx >= n)
        return;

    int x = idx % mapSize;
    int y = idx / mapSize;

    // outgoing
    float outN = fluxN[idx];
    float outS = fluxS[idx];
    float outE = fluxE[idx];
    float outW = fluxW[idx];
    float outSum = outN + outS + outE + outW;

    // incoming from neighbors (their outgoing toward this cell)
    float inN = (y > 0) ? fluxS[cellId(x, y - 1, mapSize)] : 0.0f;           // neighbor north's south flux
    float inS = (y < mapSize - 1) ? fluxN[cellId(x, y + 1, mapSize)] : 0.0f; // neighbor south's north flux
    float inE = (x < mapSize - 1) ? fluxW[cellId(x + 1, y, mapSize)] : 0.0f; // neighbor east's west flux
    float inW = (x > 0) ? fluxE[cellId(x - 1, y, mapSize)] : 0.0f;           // neighbor west's east flux
    float inSum = inN + inS + inE + inW;

    // simple update: newWater = oldWater + inflows - outflows
    water[idx] = water[idx] + inSum - outSum;
    // clamp to non-negative
    if (water[idx] < 0.0f)
        water[idx] = 0.0f;
}

// add small rainfall to each cell to drive flow (per iteration)
static __global__ void addRainfall(float *water, int n, float amount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    water[idx] += amount;
}

// Transport sediment and erode/deposit based on flow velocity using brush-weighted updates
static __global__ void transportSediment(float *height, float *water, float *sediment, const float *fluxN, const float *fluxS, const float *fluxE, const float *fluxW, int mapSize, float capacityFactor, float erodeSpeed, float depositSpeed, float evaporate, const int *brushIndexOffsets, const float *brushWeights, int brushLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = mapSize * mapSize;
    if (idx >= n)
        return;

    float outN = fluxN[idx];
    float outS = fluxS[idx];
    float outE = fluxE[idx];
    float outW = fluxW[idx];
    float flow = outN + outS + outE + outW;

    // approximate velocity as flow / (water + small_eps)
    float w = water[idx];
    const float eps = 1e-6f;
    float velocity = flow / (w + eps);

    // compute local slope (height relative to neighbors)
    int x = idx % mapSize;
    int y = idx / mapSize;
    float h = height[idx];
    float hN = (y > 0) ? height[cellId(x, y - 1, mapSize)] : h;
    float hS = (y < mapSize - 1) ? height[cellId(x, y + 1, mapSize)] : h;
    float hE = (x < mapSize - 1) ? height[cellId(x + 1, y, mapSize)] : h;
    float hW = (x > 0) ? height[cellId(x - 1, y, mapSize)] : h;
    float avgNeighbor = 0.25f * (hN + hS + hE + hW);
    float slope = fmaxf(0.0f, h - avgNeighbor);

    float capacity = capacityFactor * velocity * w * (1.0f + slope) + 1e-6f; // slope amplifies capacity

    // simple sediment advection: send a fraction of local sediment along outgoing fluxes
    float sLocal = sediment[idx];
    if (flow > eps && sLocal > 0.0f)
    {
        float sendFactor = 0.8f; // fraction of sediment moved per iteration
        float sendN = sLocal * (outN / flow) * sendFactor;
        float sendS = sLocal * (outS / flow) * sendFactor;
        float sendE = sLocal * (outE / flow) * sendFactor;
        float sendW = sLocal * (outW / flow) * sendFactor;
        float sent = 0.0f;
        if (y > 0)
        {
            atomicAdd(&sediment[cellId(x, y - 1, mapSize)], sendN);
            sent += sendN;
        }
        if (y < mapSize - 1)
        {
            atomicAdd(&sediment[cellId(x, y + 1, mapSize)], sendS);
            sent += sendS;
        }
        if (x < mapSize - 1)
        {
            atomicAdd(&sediment[cellId(x + 1, y, mapSize)], sendE);
            sent += sendE;
        }
        if (x > 0)
        {
            atomicAdd(&sediment[cellId(x - 1, y, mapSize)], sendW);
            sent += sendW;
        }
        // remove sent sediment from this cell
        if (sent > 0.0f)
            atomicAdd(&sediment[idx], -sent);
        sLocal -= sent;
    }

    float s = sLocal;

    if (s > capacity)
    {
        // deposit: prefer downstream cell to form channel deposits
        float amount = (s - capacity) * depositSpeed;
        if (amount > 0.0f)
        {
            // find dominant outflow direction
            float maxOut = outN;
            int dx = 0, dy = -1; // north
            if (outS > maxOut)
            {
                maxOut = outS;
                dx = 0;
                dy = 1;
            }
            if (outE > maxOut)
            {
                maxOut = outE;
                dx = 1;
                dy = 0;
            }
            if (outW > maxOut)
            {
                maxOut = outW;
                dx = -1;
                dy = 0;
            }

            int tx = x + dx;
            int ty = y + dy;
            if (tx >= 0 && tx < mapSize && ty >= 0 && ty < mapSize)
            {
                int tid = cellId(tx, ty, mapSize);
                atomicAdd(&height[tid], amount);
            }
            else
            {
                atomicAdd(&height[idx], amount);
            }
            atomicAdd(&sediment[idx], -amount);
        }
    }
    else
    {
        // erode focused along dominant flow direction and sides to form channels/ridges
        float amount = fminf((capacity - s) * erodeSpeed, height[idx] * 0.5f);
        if (amount > 0.0f)
        {
            // dominant direction
            float maxOut = outN;
            int dx = 0, dy = -1; // north
            if (outS > maxOut)
            {
                maxOut = outS;
                dx = 0;
                dy = 1;
            }
            if (outE > maxOut)
            {
                maxOut = outE;
                dx = 1;
                dy = 0;
            }
            if (outW > maxOut)
            {
                maxOut = outW;
                dx = -1;
                dy = 0;
            }

            // sides (perp directions)
            int leftX = -dy, leftY = dx;
            int rightX = dy, rightY = -dx;

            float centerFrac = 0.6f;
            float sideFrac = 0.4f;

            float centerErode = amount * centerFrac;
            float sideErode = amount * sideFrac * 0.5f; // each side

            // center
            atomicAdd(&height[idx], -centerErode);

            // left side
            int lx = x + leftX;
            int ly = y + leftY;
            if (lx >= 0 && lx < mapSize && ly >= 0 && ly < mapSize)
            {
                atomicAdd(&height[cellId(lx, ly, mapSize)], -sideErode);
            }

            // right side
            int rx = x + rightX;
            int ry = y + rightY;
            if (rx >= 0 && rx < mapSize && ry >= 0 && ry < mapSize)
            {
                atomicAdd(&height[cellId(rx, ry, mapSize)], -sideErode);
            }

            // add eroded sediment to local sediment
            atomicAdd(&sediment[idx], centerErode + 2.0f * sideErode);
        }
    }

    // evaporate water
    water[idx] = w * (1.0f - evaporate);
}

void GpuGB::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize)
{
    const int N = mapSize * mapSize;
    const size_t bytes = N * sizeof(float);

    float *d_height = nullptr;
    float *d_water = nullptr;
    float *d_sediment = nullptr;
    float *d_fluxN = nullptr, *d_fluxS = nullptr, *d_fluxE = nullptr, *d_fluxW = nullptr;
    int *d_brushIndices = nullptr;
    float *d_brushWeights = nullptr;

    // allocate
    CHECK_CUDA(cudaMalloc(&d_height, bytes));
    CHECK_CUDA(cudaMalloc(&d_water, bytes));
    CHECK_CUDA(cudaMalloc(&d_sediment, bytes));
    CHECK_CUDA(cudaMalloc(&d_fluxN, bytes));
    CHECK_CUDA(cudaMalloc(&d_fluxS, bytes));
    CHECK_CUDA(cudaMalloc(&d_fluxE, bytes));
    CHECK_CUDA(cudaMalloc(&d_fluxW, bytes));

    // copy initial height
    CHECK_CUDA(cudaMemcpy(d_height, heightmap.data(), bytes, cudaMemcpyHostToDevice));

    // initialize water and sediment
    std::vector<float> initWater(N, p.startWater);
    std::vector<float> initSed(N, 0.0f);
    CHECK_CUDA(cudaMemcpy(d_water, initWater.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sediment, initSed.data(), bytes, cudaMemcpyHostToDevice));

    // generate brush and copy to device
    std::vector<int> brushIndexOffsets;
    std::vector<float> brushWeights;
    generateBrush(p.brushRadius, brushIndexOffsets, brushWeights, mapSize);
    int brushLen = static_cast<int>(brushIndexOffsets.size());
    if (brushLen > 0)
    {
        CHECK_CUDA(cudaMalloc(&d_brushIndices, brushLen * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_brushWeights, brushLen * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_brushIndices, brushIndexOffsets.data(), brushLen * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_brushWeights, brushWeights.data(), brushLen * sizeof(float), cudaMemcpyHostToDevice));
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // parameters for flux computation
    float flowFactor = 0.5f * p.erodeSpeed; // controls speed of flow between cells
    int iterations = p.maxLifetime;         // reuse maxLifetime as number of solver iterations

    for (int it = 0; it < iterations; ++it)
    {
        // add a little rain each iteration to drive flow
        float rainAmount = p.startWater * 0.001f;
        addRainfall<<<blocks, threads>>>(d_water, N, rainAmount);

        computeOutFluxes<<<blocks, threads>>>(d_height, d_water, d_fluxN, d_fluxS, d_fluxE, d_fluxW, mapSize, flowFactor);
        applyFluxes<<<blocks, threads>>>(d_water, d_fluxN, d_fluxS, d_fluxE, d_fluxW, mapSize);
        transportSediment<<<blocks, threads>>>(d_height, d_water, d_sediment, d_fluxN, d_fluxS, d_fluxE, d_fluxW, mapSize, p.sedimentCapacityFactor, p.erodeSpeed, p.depositSpeed, p.evaporateSpeed, d_brushIndices, d_brushWeights, brushLen);
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // copy back height
    CHECK_CUDA(cudaMemcpy(heightmap.data(), d_height, bytes, cudaMemcpyDeviceToHost));

    // free
    CHECK_CUDA(cudaFree(d_height));
    CHECK_CUDA(cudaFree(d_water));
    CHECK_CUDA(cudaFree(d_sediment));
    CHECK_CUDA(cudaFree(d_fluxN));
    CHECK_CUDA(cudaFree(d_fluxS));
    CHECK_CUDA(cudaFree(d_fluxE));
    CHECK_CUDA(cudaFree(d_fluxW));
    if (d_brushIndices)
        cudaFree(d_brushIndices);
    if (d_brushWeights)
        cudaFree(d_brushWeights);
}