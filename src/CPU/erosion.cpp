#include "cpu.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <iostream>

static inline float clampf(float v, float a, float b)
{
    return v < a ? a : (v > b ? b : v);
}

struct particle
{
    float posX;
    float posY;
    float dirX;
    float dirY;
    float speed;
    float water;
    float sediment;
};

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

void calculateHeightAndGradient(const std::vector<float> &heightmap, int mapSize, float posX, float posY, float &outHeight, float &outGradX, float &outGradY)
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

void Cpu::applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize)
{
    std::vector<int> brushIndexOffsets;
    std::vector<float> brushWeights;
    generateBrush(p.brushRadius, brushIndexOffsets, brushWeights, mapSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_int_distribution<> distr(p.brushRadius, mapSize + p.brushRadius);
    std::uniform_int_distribution<> distr(0, mapSize - 1);

    std::vector<int> randomIndices(p.numDrops);
    for (int i = 0; i < p.numDrops; ++i)
    {
        randomIndices[i] = distr(gen) * mapSize + distr(gen);
    }

    for (int drop = 0; drop < p.numDrops; ++drop)
    {
        int nodeIndex = randomIndices[drop];
        float posX = float(nodeIndex % mapSize);
        float posY = float(nodeIndex) / mapSize;

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
            calculateHeightAndGradient(heightmap, mapSize, posX, posY, height, gradX, gradY);
            // update direction and move
            dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
            dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);
            // normalize direction
            float len = std::max(0.01f, std::sqrt(dirX * dirX + dirY * dirY));
            dirX /= len;
            dirY /= len;
            posX += dirX;
            posY += dirY;

            // stop sim if droplet is outside bounds
            if ((dirX == 0 && dirY == 0) || posX < 0 || posX >= mapSize - 1 || posY < 0 || posY >= mapSize - 1)
                break;
            // compute new height
            float newHeight, newGradX, newGradY;
            calculateHeightAndGradient(heightmap, mapSize, posX, posY, newHeight, newGradX, newGradY);
            float deltaHeight = newHeight - height;

            // compute sediment capacity
            float sedimentCapacity = std::max(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);
            // if carrying more sediment than capacity, , or if flowing uphill:
            if (sediment > sedimentCapacity || deltaHeight > 0.0f)
            {
                // deposit sediment
                float amountToDeposit = (deltaHeight > 0.0f) ? std::min(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
                sediment -= amountToDeposit;
                // deposit using bilinear interpolation
                heightmap[dropletIndex] += amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY);
                heightmap[dropletIndex + 1] += amountToDeposit * cellOffsetX * (1 - cellOffsetY);
                heightmap[dropletIndex + mapSize] += amountToDeposit * (1 - cellOffsetX) * cellOffsetY;
                heightmap[dropletIndex + mapSize + 1] += amountToDeposit * cellOffsetX * cellOffsetY;
            }
            else
            {
                // erode using brush
                float amountToErode = std::min((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
                for (size_t bi = 0; bi < brushIndexOffsets.size(); bi++)
                {
                    int erodeIndex = dropletIndex + brushIndexOffsets[bi];
                    if (erodeIndex < 0 || erodeIndex >= (int)heightmap.size())
                        continue; // skip out-of-bounds
                    float weightedErodeAmount = amountToErode * brushWeights[bi];
                    float deltaSediment = (heightmap[erodeIndex] < weightedErodeAmount) ? heightmap[erodeIndex] : weightedErodeAmount;
                    heightmap[erodeIndex] -= deltaSediment;
                    sediment += deltaSediment;
                }
            }
            // update speed & water (use compute-shader style)
            speed = std::sqrt(std::max(0.0f, speed * speed + deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);
        }
    }
}
