#include "gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

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

__device__ __forceinline__ int globalThreadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ void calculateHeightAndGradient(float *heightmap, int mapSize, float posX, float posY, float &outHeight, float &outGradX, float &outGradY)
{
    int nodeX = (int)posX;
    int nodeY = (int)posY;

    // clamp to valid range so we can safely sample neighbours (nodeX/nodeY up to mapSize-2)
    if (nodeX < 0)
        nodeX = 0;
    if (nodeY < 0)
        nodeY = 0;
    if (nodeX > mapSize - 2)
        nodeX = mapSize - 2;
    if (nodeY > mapSize - 2)
        nodeY = mapSize - 2;

    int dropletIndex = nodeY * mapSize + nodeX;

    // compute droplet's offset inside the cell (0-1)
    float x = posX - float(nodeX);
    float y = posY - float(nodeY);

    // compute height and gradient using bilinear interpolation of surrounding heights
    float heightNW = heightmap[dropletIndex];
    float heightNE = heightmap[dropletIndex + 1];
    float heightSW = heightmap[dropletIndex + mapSize];
    float heightSE = heightmap[dropletIndex + mapSize + 1];

    // calculate droplet's height
    outHeight = heightNW * (1 - x) * (1 - y) + heightNE * x * (1 - y) + heightSW * (1 - x) * y + heightSE * x * y;

    // compute gradient
    outGradX = (heightNE - heightNW) * (1 - y) + (heightSE - heightSW) * y;
    outGradY = (heightSW - heightNW) * (1 - x) + (heightSE - heightNE) * x;
}

__global__ void initRNG(curandState *states, unsigned int seed, int numDrops)
{
    int id = globalThreadId();
    if (id >= numDrops)
        return;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void erosionKernel(float *d_map, const ErosionParams &p, int mapSize, curandState *states, const int *brushIndexOffsets, const float *brushWeights, int brushLength, int numDrops)
{
    int id = globalThreadId();
    if (id >= numDrops)
        return;

    curandState localState = states[id];
    float r = curand_uniform(&localState);

    int nodeIndex = (int)(curand_uniform(&localState) * (mapSize * mapSize - 1));
    float posX = (float)(nodeIndex % mapSize);
    float posY = (float)(nodeIndex / mapSize);

    float dirX = 0.0f, dirY = 0.0f;
    float speed = p.startSpeed;
    float water = p.startWater;
    float sediment = 0.0f;

    for (int lifetime = 0; lifetime < p.maxLifetime; lifetime++)
    {
        int nodeX = (int)posX;
        int nodeY = (int)posY;
        int dropletIndex = nodeY * mapSize + nodeX;

        // compute droplet's offset inside the cell (0-1)
        float cellOffsetX = posX - float(nodeX);
        float cellOffsetY = posY - float(nodeY);
        // compute height and gradient at current position
        float height, gradX, gradY;
        calculateHeightAndGradient(d_map, mapSize, posX, posY, height, gradX, gradY);
        // update direction and move
        dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
        dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);
        // normalize direction
        float len = fmaxf(0.01f, sqrtf(dirX * dirX + dirY * dirY));
        dirX /= len;
        dirY /= len;
        posX += dirX;
        posY += dirY;

        // stop sim if droplet is outside bounds
        if ((dirX == 0 && dirY == 0) || posX < 0 || posX >= mapSize - 1 || posY < 0 || posY >= mapSize - 1)
            break;
        // compute new height
        float newHeight, newGradX, newGradY;
        calculateHeightAndGradient(d_map, mapSize, posX, posY, newHeight, newGradX, newGradY);
        float deltaHeight = newHeight - height;

        // compute sediment capacity
        float sedimentCapacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);
        // if carrying more sediment than capacity, , or if flowing uphill:
        if (sediment > sedimentCapacity || deltaHeight > 0.0f)
        {
            // deposit sediment
            float amountToDeposit = (deltaHeight > 0.0f) ? fminf(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
            sediment -= amountToDeposit;
            // deposit using bilinear interpolation
            if (nodeX < mapSize - 1 && nodeY < mapSize - 1)
            {
                atomicAdd(&d_map[dropletIndex], amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY));
                atomicAdd(&d_map[dropletIndex + 1], amountToDeposit * cellOffsetX * (1 - cellOffsetY));
                atomicAdd(&d_map[dropletIndex + mapSize], amountToDeposit * (1 - cellOffsetX) * cellOffsetY);
                atomicAdd(&d_map[dropletIndex + mapSize + 1], amountToDeposit * cellOffsetX * cellOffsetY);
            }
            else
            {
                break;
            }
        }
        else
        {
            // erode using brush
            float amountToErode = fminf((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
            for (size_t bi = 0; bi < brushLength; bi++)
            {
                int erodeIndex = dropletIndex + brushIndexOffsets[bi];
                if (erodeIndex < 0 || erodeIndex >= (mapSize * mapSize))
                    continue; // skip out-of-bounds
                float weightedErodeAmount = amountToErode * brushWeights[bi];
                float deltaSediment = (d_map[erodeIndex] < weightedErodeAmount) ? d_map[erodeIndex] : weightedErodeAmount;
                atomicAdd(&d_map[erodeIndex], -deltaSediment);
                sediment += deltaSediment;
            }
        }
        speed = sqrtf(fmaxf(0.0f, speed * speed + deltaHeight * p.gravity));
        water *= (1.0f - p.evaporateSpeed);
    }
    states[id] = localState;
}

void Gpu::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize)
{
    const size_t mapBytes = mapSize * mapSize * sizeof(float);
    float *d_map = nullptr, *d_brushWeights = nullptr;
    int *d_brushIndices = nullptr;

    std::vector<int> brushIndexOffsets;
    std::vector<float> brushWeights;
    generateBrush(p.brushRadius, brushIndexOffsets, brushWeights, mapSize);

    CHECK_CUDA(cudaMalloc(&d_map, mapBytes));
    CHECK_CUDA(cudaMemcpy(d_map, heightmap.data(), mapBytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_brushWeights, brushWeights.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_brushWeights, brushWeights.data(), brushWeights.size() * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_brushIndices, brushIndexOffsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_brushIndices, brushIndexOffsets.data(), brushIndexOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (p.numDrops + threads - 1) / threads;

    curandState *d_states;
    cudaMalloc(&d_states, p.numDrops * sizeof(curandState));

    initRNG<<<blocks, threads>>>(d_states, time(NULL), p.numDrops);

    for (int drop = 0; drop < p.numDrops; ++drop)
    {
        erosionKernel<<<blocks, threads>>>(d_map, p, mapSize, d_states, d_brushIndices, d_brushWeights, brushIndexOffsets.size(), p.numDrops);
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(heightmap.data(), d_map, mapBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_map));
}