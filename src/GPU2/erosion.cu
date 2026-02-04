#include "gpu.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>

static constexpr int THREADS = 256;
static constexpr int DROPS_PER_THREAD = 8;

// static member definitions
float *Gpu2::d_map = nullptr;
size_t Gpu2::d_mapBytes = 0;

int *Gpu2::d_brushIndices = nullptr;
float *Gpu2::d_brushWeights = nullptr;
int Gpu2::brushLength = 0;
int Gpu2::brushMapSize = 0;

curandState *Gpu2::d_rngStates = nullptr;
int Gpu2::rngStateCount = 0;

float *Gpu2::h_pinnedMap = nullptr;
size_t Gpu2::h_pinnedBytes = 0;

cudaStream_t Gpu2::stream = nullptr;

static void generateBrush(int radius, int mapSize, std::vector<int> &offsets, std::vector<float> &weights)
{
    offsets.clear();
    weights.clear();

    float sum = 0.f;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            float d = sqrtf(float(x * x + y * y));
            if (d <= radius)
            {
                offsets.push_back(y * mapSize + x);
                float w = 1.f - d / radius;
                weights.push_back(w);
                sum += w;
            }
        }
    }

    for (float &w : weights)
        w /= sum;
}

void Gpu2::ensureStream()
{
    if (!stream)
        CHECK_CUDA(cudaStreamCreate(&stream));
}

void Gpu2::uploadBrush(const ErosionParams &p, int mapSize)
{
    std::vector<int> offsets;
    std::vector<float> weights;
    generateBrush(p.brushRadius, mapSize, offsets, weights);

    // re-create brush buffers if brush size changed or if mapSize changed
    if ((int)offsets.size() != brushLength || mapSize != brushMapSize)
    {
        if (d_brushIndices)
            cudaFree(d_brushIndices);
        if (d_brushWeights)
            cudaFree(d_brushWeights);

        brushLength = (int)offsets.size();
        brushMapSize = mapSize;

        CHECK_CUDA(cudaMalloc(&d_brushIndices,
                              brushLength * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_brushWeights,
                              brushLength * sizeof(float)));

        CHECK_CUDA(cudaMemcpyAsync(
            d_brushIndices, offsets.data(),
            brushLength * sizeof(int),
            cudaMemcpyHostToDevice, stream));

        CHECK_CUDA(cudaMemcpyAsync(
            d_brushWeights, weights.data(),
            brushLength * sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }
}

static __device__ __forceinline__ int globalThreadId()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

static __device__ void calculateHeightAndGradient(float *heightmap, int mapSize, float posX, float posY, float &outHeight, float &outGradX, float &outGradY)
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

static __global__ void initRNG(curandState *states, unsigned int seed, int numStates)
{
    int id = globalThreadId();
    if (id < numStates)
        curand_init(seed, id, 0, &states[id]);
}

static __global__ void erosionKernel(float *d_map, ErosionParams p, int mapSize, curandState *states, const int *brushIndexOffsets, const float *brushWeights, int brushLength, int numDrops, int dropsPerThread, int tilesPerRow)
{
    int id = globalThreadId();
    int totalThreads = gridDim.x * blockDim.x;
    if (id >= totalThreads)
        return;

    curandState localState = states[id];

    // tile this block works on (tilesPerRow x tilesPerRow grid)

    int tileX = blockIdx.x % tilesPerRow;
    int tileY = blockIdx.x / tilesPerRow;

    int tileSize = (mapSize - 1 + tilesPerRow - 1) / tilesPerRow; // ceil div
    int tileMinX = tileX * tileSize;
    int tileMinY = tileY * tileSize;

    // compute global drop index range this thread should handle
    int threadStartDrop = id * dropsPerThread;

    for (int d = 0; d < dropsPerThread; d++)
    {
        int globalDropIdx = threadStartDrop + d;
        if (globalDropIdx >= numDrops)
            break;

        // pick a start within the block's tile to improve locality
        float rx = curand_uniform(&localState);
        float ry = curand_uniform(&localState);

        int tileMaxX = tileMinX + tileSize;
        int tileMaxY = tileMinY + tileSize;
        if (tileMaxX > mapSize - 1)
            tileMaxX = mapSize - 1;
        if (tileMaxY > mapSize - 1)
            tileMaxY = mapSize - 1;
        // sample uniformly inside the tile bounds (may be degenerate for very small tiles)
        float posX = tileMinX + rx * float(tileMaxX - tileMinX);
        float posY = tileMinY + ry * float(tileMaxY - tileMinY);

        // clamp to safe bilinear sampling region
        posX = fminf(posX, mapSize - 2.0001f);
        posY = fminf(posY, mapSize - 2.0001f);

        float dirX = 0.0f, dirY = 0.0f;
        float speed = p.startSpeed;
        float water = p.startWater;
        float sediment = 0.0f;

        for (int lifetime = 0; lifetime < p.maxLifetime; lifetime++)
        {
            int nodeX = (int)posX;
            int nodeY = (int)posY;
            int dropletIndex = nodeY * mapSize + nodeX;

            // droplet offset in cell
            float cellOffsetX = posX - float(nodeX);
            float cellOffsetY = posY - float(nodeY);

            // compute height & gradient
            float height, gradX, gradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, height, gradX, gradY);

            // update direction
            dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
            dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);
            float len = fmaxf(0.01f, sqrtf(dirX * dirX + dirY * dirY));
            dirX /= len;
            dirY /= len;
            posX += dirX;
            posY += dirY;

            // stop if outside safe bounds for bilinear interpolation
            if (posX < 0.0f || posX >= mapSize - 1 || posY < 0.0f || posY >= mapSize - 1)
                break;

            // compute new height & delta
            float newHeight, newGradX, newGradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, newHeight, newGradX, newGradY);
            float deltaHeight = newHeight - height;

            // sediment capacity
            float sedimentCapacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);

            if (sediment > sedimentCapacity || deltaHeight > 0.0f)
            {
                // deposit sediment safely
                int safeX = nodeX;
                int safeY = nodeY;
                if (safeX >= mapSize - 1)
                    safeX = mapSize - 2;
                if (safeY >= mapSize - 1)
                    safeY = mapSize - 2;

                int safeIndex = safeY * mapSize + safeX;
                float amountToDeposit = (deltaHeight > 0.0f) ? fminf(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
                sediment -= amountToDeposit;

                atomicAdd(&d_map[safeIndex], amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY));
                atomicAdd(&d_map[safeIndex + 1], amountToDeposit * cellOffsetX * (1 - cellOffsetY));
                atomicAdd(&d_map[safeIndex + mapSize], amountToDeposit * (1 - cellOffsetX) * cellOffsetY);
                atomicAdd(&d_map[safeIndex + mapSize + 1], amountToDeposit * cellOffsetX * cellOffsetY);
            }
            else
            {
                // erosion using brush safely: use linear index offsets to avoid negative-division issues
                float amountToErode = fminf((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
                for (int bi = 0; bi < brushLength; bi++)
                {
                    int offset = brushIndexOffsets[bi];
                    int erodeIndex = dropletIndex + offset;

                    // bounds check on linear index
                    if (erodeIndex < 0 || erodeIndex >= mapSize * mapSize)
                        continue;

                    float weightedErodeAmount = amountToErode * brushWeights[bi];
                    float deltaSediment = fminf(d_map[erodeIndex], weightedErodeAmount);

                    atomicAdd(&d_map[erodeIndex], -deltaSediment);
                    sediment += deltaSediment;
                }
            }

            speed = sqrtf(fmaxf(0.0f, speed * speed + deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);
        }
    }

    // save RNG state
    states[id] = localState;
}

void Gpu2::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize)
{
    ensureStream();

    size_t bytes = mapSize * mapSize * sizeof(float);
    ensurePinnedHost(bytes);
    ensureDeviceMap(bytes);

    std::memcpy(h_pinnedMap, heightmap.data(), bytes);

    CHECK_CUDA(cudaMemcpyAsync(d_map, h_pinnedMap, bytes, cudaMemcpyHostToDevice, stream));

    uploadBrush(p, mapSize);

    int dropsPerThread = DROPS_PER_THREAD;
    int threadsNeeded = (p.numDrops + dropsPerThread - 1) / dropsPerThread;

    int blocks = (threadsNeeded + THREADS - 1) / THREADS;
    int totalThreads = blocks * THREADS;

    ensureRNG(totalThreads);

    int tilesPerRow = (int)ceilf(sqrtf((float)blocks));

    erosionKernel<<<blocks, THREADS, 0, stream>>>(
        d_map, p, mapSize,
        d_rngStates,
        d_brushIndices,
        d_brushWeights,
        brushLength,
        p.numDrops,
        dropsPerThread,
        tilesPerRow);

    CHECK_CUDA(cudaMemcpyAsync(h_pinnedMap, d_map, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::memcpy(heightmap.data(), h_pinnedMap, bytes);
}

void Gpu2::ensurePinnedHost(size_t bytes)
{
    if (h_pinnedBytes < bytes)
    {
        if (h_pinnedMap)
            CHECK_CUDA(cudaFreeHost(h_pinnedMap));

        CHECK_CUDA(cudaMallocHost(&h_pinnedMap, bytes));
        h_pinnedBytes = bytes;
    }
}

void Gpu2::ensureDeviceMap(size_t bytes)
{
    if (d_mapBytes < bytes)
    {
        if (d_map)
            CHECK_CUDA(cudaFree(d_map));

        CHECK_CUDA(cudaMalloc(&d_map, bytes));
        d_mapBytes = bytes;
    }
}

void Gpu2::ensureRNG(int totalThreads)
{
    if (rngStateCount < totalThreads)
    {
        if (d_rngStates)
            CHECK_CUDA(cudaFree(d_rngStates));

        rngStateCount = totalThreads;
        CHECK_CUDA(cudaMalloc(&d_rngStates,
                              rngStateCount * sizeof(curandState)));

        int blocks = (rngStateCount + THREADS - 1) / THREADS;
        initRNG<<<blocks, THREADS, 0, stream>>>(
            d_rngStates, (unsigned)time(nullptr), rngStateCount);
    }
}

void Gpu2::shutdown()
{
    if (d_map)
        cudaFree(d_map);
    if (d_brushIndices)
        cudaFree(d_brushIndices);
    if (d_brushWeights)
        cudaFree(d_brushWeights);
    if (d_rngStates)
        cudaFree(d_rngStates);
    if (h_pinnedMap)
        cudaFreeHost(h_pinnedMap);
    if (stream)
        cudaStreamDestroy(stream);

    d_map = nullptr;
    d_brushIndices = nullptr;
    d_brushWeights = nullptr;
    d_rngStates = nullptr;
    h_pinnedMap = nullptr;
    stream = nullptr;
}