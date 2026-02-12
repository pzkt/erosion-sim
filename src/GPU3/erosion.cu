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

__device__ __forceinline__ int ceilDiv(int a, int b)
{
    return (a + b - 1) / b;
}

__device__ __forceinline__ bool inTile(int x, int y, int tileMinX, int tileMinY, int tileWidth, int tileHeight)
{
    return x >= tileMinX && x < tileMinX + tileWidth && y >= tileMinY && y < tileMinY + tileHeight;
}

__device__ __forceinline__ void addHeight(float *d_map, float *s_tile, int mapSize, int x, int y, float value, int tileMinX, int tileMinY, int tileWidth, int tileHeight)
{
    if (inTile(x, y, tileMinX, tileMinY, tileWidth, tileHeight))
    {
        int localIdx = (y - tileMinY) * tileWidth + (x - tileMinX);
        atomicAdd(&s_tile[localIdx], value);
    }
    else
    {
        atomicAdd(&d_map[y * mapSize + x], value);
    }
}

static __device__ void calculateHeightAndGradient(float *heightmap, int mapSize, float posX, float posY, float &outHeight, float &outGradX, float &outGradY, float *s_tile, int tileMinX, int tileMinY, int tileWidth)
{
    int nodeX = max(0, min((int)posX, mapSize - 2));
    int nodeY = max(0, min((int)posY, mapSize - 2));

    float x = posX - nodeX;
    float y = posY - nodeY;

    int idxNW = nodeY * mapSize + nodeX;
    int idxNE = idxNW + 1;
    int idxSW = idxNW + mapSize;
    int idxSE = idxSW + 1;

    auto sample = [&](int ix, int iy, int idx)
    {
        float h = heightmap[idx];

        if (inTile(ix, iy, tileMinX, tileMinY, tileWidth, tileWidth))
        {
            int localX = ix - tileMinX;
            int localY = iy - tileMinY;
            h += s_tile[localY * tileWidth + localX];
        }

        return h;
    };

    float heightNW = sample(nodeX, nodeY, idxNW);
    float heightNE = sample(nodeX + 1, nodeY, idxNE);
    float heightSW = sample(nodeX, nodeY + 1, idxSW);
    float heightSE = sample(nodeX + 1, nodeY + 1, idxSE);

    float oneMinusX = 1.0f - x;
    float oneMinusY = 1.0f - y;

    outHeight = heightNW * oneMinusX * oneMinusY + heightNE * x * oneMinusY + heightSW * oneMinusX * y + heightSE * x * y;
    outGradX = (heightNE - heightNW) * oneMinusY + (heightSE - heightSW) * y;
    outGradY = (heightSW - heightNW) * oneMinusX + (heightSE - heightNE) * x;
}

static __global__ void initRNG(curandState *states, unsigned int seed, int numStates)
{
    int id = globalThreadId();
    if (id < numStates)
        curand_init(seed, id, 0, &states[id]);
}

static __global__ void erosionKernel(float *d_map, ErosionParams p, int mapSize, curandState *states, const int *brushIndexOffsets, const float *brushWeights, int brushLength, int numDrops, int tilesPerRow)
{
    int id = globalThreadId();

    curandState localState = states[id];

    // Tile setup

    int tileX = blockIdx.x % tilesPerRow;
    int tileY = blockIdx.x / tilesPerRow;

    int tileSize = ceilDiv(mapSize - 1, tilesPerRow);

    int tileMinX = tileX * tileSize;
    int tileMinY = tileY * tileSize;

    int tileWidth = min(tileSize + 1, mapSize - tileMinX);
    int tileHeight = min(tileSize + 1, mapSize - tileMinY);

    int tileArea = tileWidth * tileHeight;

    extern __shared__ float s_tile[];

    // Zero shared tile
    for (int i = threadIdx.x; i < tileArea; i += blockDim.x)
        s_tile[i] = 0.0f;

    __syncthreads();

    for (int d = 0; d < DROPS_PER_THREAD; d++)
    {
        float rx = curand_uniform(&localState);
        float ry = curand_uniform(&localState);

        int tileMaxX = min(tileMinX + tileSize, mapSize - 1);
        int tileMaxY = min(tileMinY + tileSize, mapSize - 1);

        float posX = tileMinX + rx * (tileMaxX - tileMinX);
        float posY = tileMinY + ry * (tileMaxY - tileMinY);

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

            float cellOffsetX = posX - nodeX;
            float cellOffsetY = posY - nodeY;

            float height, gradX, gradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, height, gradX, gradY, s_tile, tileMinX, tileMinY, tileWidth);

            // Update direction
            dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
            dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);

            float invLen = rsqrtf(dirX * dirX + dirY * dirY + 1e-6f);
            dirX *= invLen;
            dirY *= invLen;

            posX += dirX;
            posY += dirY;

            if (!inTile((int)posX, (int)posY, tileMinX, tileMinY, tileWidth, tileHeight))
                break;

            float newHeight, newGradX, newGradY;
            calculateHeightAndGradient(d_map, mapSize, posX, posY, newHeight, newGradX, newGradY, s_tile, tileMinX, tileMinY, tileWidth);
            float deltaHeight = newHeight - height;
            float sedimentCapacity = fmaxf(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);

            // deposit
            if (sediment > sedimentCapacity || deltaHeight > 0.0f)
            {
                float amountToDeposit = (deltaHeight > 0.0f) ? fminf(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
                sediment -= amountToDeposit;

                int gx0 = nodeX;
                int gy0 = nodeY;
                int gx1 = gx0 + 1;
                int gy1 = gy0 + 1;

                float w[2][2] =
                    {
                        {amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY),
                         amountToDeposit * cellOffsetX * (1 - cellOffsetY)},

                        {amountToDeposit * (1 - cellOffsetX) * cellOffsetY,
                         amountToDeposit * cellOffsetX * cellOffsetY}};

                int gx[2] = {gx0, gx1};
                int gy[2] = {gy0, gy1};

                for (int iy = 0; iy < 2; iy++)
                {
                    for (int ix = 0; ix < 2; ix++)
                    {
                        float weight = w[iy][ix];
                        if (weight == 0.0f)
                            continue;

                        int x = gx[ix];
                        int y = gy[iy];

                        if (x >= mapSize || y >= mapSize)
                            continue;

                        addHeight(d_map, s_tile, mapSize, x, y, weight, tileMinX, tileMinY, tileWidth, tileHeight);
                    }
                }
            }

            // erosion
            else
            {
                float amountToErode = fminf((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
                int dropletIndex = nodeY * mapSize + nodeX;

                for (int bi = 0; bi < brushLength; bi++)
                {
                    int erodeIndex = dropletIndex + brushIndexOffsets[bi];
                    if (erodeIndex < 0 || erodeIndex >= mapSize * mapSize)
                        continue;

                    float weightedAmount = amountToErode * brushWeights[bi];
                    int ex = erodeIndex % mapSize;
                    int ey = erodeIndex / mapSize;
                    float curVal = d_map[erodeIndex];

                    if (inTile(ex, ey, tileMinX, tileMinY, tileWidth, tileHeight))
                    {
                        int localIdx = (ey - tileMinY) * tileWidth + (ex - tileMinX);
                        curVal += s_tile[localIdx];
                    }

                    float deltaSediment = fminf(curVal, weightedAmount);
                    addHeight(d_map, s_tile, mapSize, ex, ey, -deltaSediment, tileMinX, tileMinY, tileWidth, tileHeight);
                    sediment += deltaSediment;
                }
            }
            speed = sqrtf(fmaxf(0.0f, speed * speed + deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);
        }
    }
    states[id] = localState;

    // Write back shared tile
    __syncthreads();

    for (int i = threadIdx.x; i < tileArea; i += blockDim.x)
    {
        float v = s_tile[i];
        if (v == 0.0f)
            continue;

        int lx = i % tileWidth;
        int ly = i / tileWidth;

        int gx = tileMinX + lx;
        int gy = tileMinY + ly;

        atomicAdd(&d_map[gy * mapSize + gx], v);
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

    int threadsNeeded = ceil(float(p.numDrops) / DROPS_PER_THREAD);
    int blocks = ceil(float(threadsNeeded) / THREADS);

    int tilesPerRow = ceil(sqrt(float(blocks)));
    blocks = tilesPerRow * tilesPerRow;

    int totalThreads = blocks * THREADS;

    ensureRNG(totalThreads);

    int tileSize = (mapSize - 1 + tilesPerRow - 1) / tilesPerRow;
    int maxTileWidth = tileSize + 1;
    int maxTileHeight = tileSize + 1;
    int maxTileArea = maxTileWidth * maxTileHeight;
    size_t sharedBytes = (size_t)maxTileArea * sizeof(float);

    erosionKernel<<<blocks, THREADS, (unsigned)sharedBytes, stream>>>(
        d_map, p, mapSize,
        d_rngStates,
        d_brushIndices,
        d_brushWeights,
        brushLength,
        p.numDrops,
        tilesPerRow);

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