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
float *Gpu3::d_map = nullptr;
size_t Gpu3::d_mapBytes = 0;

int *Gpu3::d_brushIndices = nullptr;
float *Gpu3::d_brushWeights = nullptr;
int Gpu3::brushLength = 0;
int Gpu3::brushMapSize = 0;

curandState *Gpu3::d_rngStates = nullptr;
int Gpu3::rngStateCount = 0;

float *Gpu3::h_pinnedMap = nullptr;
size_t Gpu3::h_pinnedBytes = 0;

cudaStream_t Gpu3::stream = nullptr;

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

void Gpu3::ensureStream()
{
    if (!stream)
        CHECK_CUDA(cudaStreamCreate(&stream));
}

void Gpu3::uploadBrush(const ErosionParams &p, int mapSize)
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

static __device__ void calculateHeightAndGradient(float *heightmap, int mapSize, float posX, float posY, float &outHeight, float &outGradX, float &outGradY, float *s_tile, int tileMinX, int tileMinY, int tileWidth, bool useShared)
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
    // fetch heights and include shared-tile contributions when available
    int idxNW = dropletIndex;
    float heightNW = heightmap[idxNW];
    if (useShared)
    {
        int iy = idxNW / mapSize;
        int ix = idxNW % mapSize;
        if (ix >= tileMinX && ix < tileMinX + tileWidth && iy >= tileMinY && iy < tileMinY + tileWidth)
        {
            int localX = ix - tileMinX;
            int localY = iy - tileMinY;
            int localIdx = localY * tileWidth + localX;
            heightNW += s_tile[localIdx];
        }
    }

    int idxNE = dropletIndex + 1;
    float heightNE = heightmap[idxNE];
    if (useShared)
    {
        int iy = idxNE / mapSize;
        int ix = idxNE % mapSize;
        if (ix >= tileMinX && ix < tileMinX + tileWidth && iy >= tileMinY && iy < tileMinY + tileWidth)
        {
            int localX = ix - tileMinX;
            int localY = iy - tileMinY;
            int localIdx = localY * tileWidth + localX;
            heightNE += s_tile[localIdx];
        }
    }

    int idxSW = dropletIndex + mapSize;
    float heightSW = heightmap[idxSW];
    if (useShared)
    {
        int iy = idxSW / mapSize;
        int ix = idxSW % mapSize;
        if (ix >= tileMinX && ix < tileMinX + tileWidth && iy >= tileMinY && iy < tileMinY + tileWidth)
        {
            int localX = ix - tileMinX;
            int localY = iy - tileMinY;
            int localIdx = localY * tileWidth + localX;
            heightSW += s_tile[localIdx];
        }
    }

    int idxSE = dropletIndex + mapSize + 1;
    float heightSE = heightmap[idxSE];
    if (useShared)
    {
        int iy = idxSE / mapSize;
        int ix = idxSE % mapSize;
        if (ix >= tileMinX && ix < tileMinX + tileWidth && iy >= tileMinY && iy < tileMinY + tileWidth)
        {
            int localX = ix - tileMinX;
            int localY = iy - tileMinY;
            int localIdx = localY * tileWidth + localX;
            heightSE += s_tile[localIdx];
        }
    }

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

static __global__ void erosionKernel(float *d_map, ErosionParams p, int mapSize, curandState *states, const int *brushIndexOffsets, const float *brushWeights, int brushLength, int numDrops, int dropsPerThread, int tilesPerRow, bool useShared)
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
    int tileWidth = tileSize + 1;
    int tileHeight = tileSize + 1;
    if (tileMinX + tileWidth > mapSize)
        tileWidth = mapSize - tileMinX;
    if (tileMinY + tileHeight > mapSize)
        tileHeight = mapSize - tileMinY;
    int tileArea = tileWidth * tileHeight;

    extern __shared__ float s_tile[]; // only valid when launch provides shared bytes
    // initialize shared tile buffer if requested and available
    if (useShared)
    {
        // zero initialize in parallel using threads of the block
        for (int i = threadIdx.x; i < tileArea; i += blockDim.x)
            s_tile[i] = 0.0f;
        __syncthreads();
    }

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

        int tileMaxX = min(tileMinX + tileSize, mapSize - 1);
        int tileMaxY = min(tileMinY + tileSize, mapSize - 1);

        float posX = tileMinX + rx * (tileMaxX - tileMinX);
        float posY = tileMinY + ry * (tileMaxY - tileMinY);

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

            // compute height & gradient (include shared-tile contributions when enabled)
            float height, gradX, gradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, height, gradX, gradY, s_tile, tileMinX, tileMinY, tileWidth, useShared);

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

            // compute new height & delta (include shared-tile contributions when enabled)
            float newHeight, newGradX, newGradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, newHeight, newGradX, newGradY, s_tile, tileMinX, tileMinY, tileWidth, useShared);
            float deltaHeight = newHeight - height;

            // sediment capacity
            float sedimentCapacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);

            if (sediment > sedimentCapacity || deltaHeight > 0.0f)
            {
                // Deposit sediment safely
                int safeX = min(nodeX, mapSize - 2);
                int safeY = min(nodeY, mapSize - 2);

                int safeIndex = safeY * mapSize + safeX;
                float amountToDeposit = (deltaHeight > 0.0f) ? fminf(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
                sediment -= amountToDeposit;

                // four bilinear contributions
                int gx0 = safeIndex % mapSize;
                int gy0 = safeIndex / mapSize;
                int gx1 = gx0 + 1;
                int gy1 = gy0 + 1;

                float w00 = amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY);
                float w10 = amountToDeposit * cellOffsetX * (1 - cellOffsetY);
                float w01 = amountToDeposit * (1 - cellOffsetX) * cellOffsetY;
                float w11 = amountToDeposit * cellOffsetX * cellOffsetY;

                // deposit to either shared tile or global map per contribution
                if (w00 != 0.0f)
                {
                    int gx = gx0;
                    int gy = gy0;
                    if (useShared && gx >= tileMinX && gx < tileMinX + tileWidth && gy >= tileMinY && gy < tileMinY + tileHeight)
                    {
                        int localIdx = (gy - tileMinY) * tileWidth + (gx - tileMinX);
                        atomicAdd(&s_tile[localIdx], w00);
                    }
                    else
                    {
                        atomicAdd(&d_map[gy * mapSize + gx], w00);
                    }
                }
                if (w10 != 0.0f && gx1 < mapSize)
                {
                    int gx = gx1;
                    int gy = gy0;
                    if (useShared && gx >= tileMinX && gx < tileMinX + tileWidth && gy >= tileMinY && gy < tileMinY + tileHeight)
                    {
                        int localIdx = (gy - tileMinY) * tileWidth + (gx - tileMinX);
                        atomicAdd(&s_tile[localIdx], w10);
                    }
                    else
                    {
                        atomicAdd(&d_map[gy * mapSize + gx], w10);
                    }
                }
                if (w01 != 0.0f && gy1 < mapSize)
                {
                    int gx = gx0;
                    int gy = gy1;
                    if (useShared && gx >= tileMinX && gx < tileMinX + tileWidth && gy >= tileMinY && gy < tileMinY + tileHeight)
                    {
                        int localIdx = (gy - tileMinY) * tileWidth + (gx - tileMinX);
                        atomicAdd(&s_tile[localIdx], w01);
                    }
                    else
                    {
                        atomicAdd(&d_map[gy * mapSize + gx], w01);
                    }
                }
                if (w11 != 0.0f && gx1 < mapSize && gy1 < mapSize)
                {
                    int gx = gx1;
                    int gy = gy1;
                    if (useShared && gx >= tileMinX && gx < tileMinX + tileWidth && gy >= tileMinY && gy < tileMinY + tileHeight)
                    {
                        int localIdx = (gy - tileMinY) * tileWidth + (gx - tileMinX);
                        atomicAdd(&s_tile[localIdx], w11);
                    }
                    else
                    {
                        atomicAdd(&d_map[gy * mapSize + gx], w11);
                    }
                }
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

                    int ey = erodeIndex / mapSize;
                    int ex = erodeIndex % mapSize;

                    // read current value including shared contribution if present
                    float curVal = d_map[erodeIndex];
                    if (useShared && ex >= tileMinX && ex < tileMinX + tileWidth && ey >= tileMinY && ey < tileMinY + tileHeight)
                    {
                        int localX = ex - tileMinX;
                        int localY = ey - tileMinY;
                        int localIdx = localY * tileWidth + localX;
                        curVal += s_tile[localIdx];
                    }

                    float deltaSediment = fminf(curVal, weightedErodeAmount);

                    if (useShared && ex >= tileMinX && ex < tileMinX + tileWidth && ey >= tileMinY && ey < tileMinY + tileHeight)
                    {
                        int localX = ex - tileMinX;
                        int localY = ey - tileMinY;
                        int localIdx = localY * tileWidth + localX;
                        atomicAdd(&s_tile[localIdx], -deltaSediment);
                    }
                    else
                    {
                        atomicAdd(&d_map[erodeIndex], -deltaSediment);
                    }

                    sediment += deltaSediment;
                }
            }

            speed = sqrtf(fmaxf(0.0f, speed * speed + deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);
        }
    }

    // save RNG state
    states[id] = localState;

    // write back shared tile into global map (only if we used shared accumulation)
    if (useShared)
    {
        __syncthreads();
        // let threads write back portions of the tile
        for (int i = threadIdx.x; i < tileArea; i += blockDim.x)
        {
            float v = s_tile[i];
            if (v != 0.0f)
            {
                int lx = i % tileWidth;
                int ly = i / tileWidth;
                int gx = tileMinX + lx;
                int gy = tileMinY + ly;
                int gidx = gy * mapSize + gx;
                // exclusive tile ensures non-atomic add is safe; use atomicAdd for portability
                atomicAdd(&d_map[gidx], v);
            }
        }
    }
}

void Gpu3::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize)
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

    // decide whether to use per-block shared-memory tile accumulation
    int tileSize = (mapSize - 1 + tilesPerRow - 1) / tilesPerRow;
    int maxTileWidth = tileSize + 1;
    int maxTileHeight = tileSize + 1;
    int maxTileArea = maxTileWidth * maxTileHeight;
    const int MAX_SHARED_BYTES = 48 * 1024;
    size_t sharedBytes = 0;
    bool useShared = false;
    if ((size_t)maxTileArea * sizeof(float) <= (size_t)MAX_SHARED_BYTES)
    {
        useShared = true;
        sharedBytes = (size_t)maxTileArea * sizeof(float);
    }

    erosionKernel<<<blocks, THREADS, (unsigned)sharedBytes, stream>>>(
        d_map, p, mapSize,
        d_rngStates,
        d_brushIndices,
        d_brushWeights,
        brushLength,
        p.numDrops,
        dropsPerThread,
        tilesPerRow,
        useShared);

    CHECK_CUDA(cudaMemcpyAsync(h_pinnedMap, d_map, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::memcpy(heightmap.data(), h_pinnedMap, bytes);
}

void Gpu3::ensurePinnedHost(size_t bytes)
{
    if (h_pinnedBytes < bytes)
    {
        if (h_pinnedMap)
            CHECK_CUDA(cudaFreeHost(h_pinnedMap));

        CHECK_CUDA(cudaMallocHost(&h_pinnedMap, bytes));
        h_pinnedBytes = bytes;
    }
}

void Gpu3::ensureDeviceMap(size_t bytes)
{
    if (d_mapBytes < bytes)
    {
        if (d_map)
            CHECK_CUDA(cudaFree(d_map));

        CHECK_CUDA(cudaMalloc(&d_map, bytes));
        d_mapBytes = bytes;
    }
}

void Gpu3::ensureRNG(int totalThreads)
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

void Gpu3::shutdown()
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