// Erosion.cpp — Sebastian Lague–style, spike-safe implementation

#include "Erosion.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cfloat>

static inline float clampf(float v, float a, float b)
{
    return v < a ? a : (v > b ? b : v);
}

// Bilinear height + gradient sampling
static void sampleHeightAndGradient(
    const std::vector<float> &map,
    int size,
    float x,
    float y,
    float &height,
    float &gradX,
    float &gradY)
{
    int xi = (int)x;
    int yi = (int)y;

    float xf = x - xi;
    float yf = y - yi;

    int x0 = std::clamp(xi, 0, size - 1);
    int x1 = std::clamp(xi + 1, 0, size - 1);
    int y0 = std::clamp(yi, 0, size - 1);
    int y1 = std::clamp(yi + 1, 0, size - 1);

    float h00 = map[y0 * size + x0];
    float h10 = map[y0 * size + x1];
    float h01 = map[y1 * size + x0];
    float h11 = map[y1 * size + x1];

    // Height
    float h0 = h00 + (h10 - h00) * xf;
    float h1 = h01 + (h11 - h01) * xf;
    height = h0 + (h1 - h0) * yf;

    // Gradient
    gradX = (h10 - h00) * (1 - yf) + (h11 - h01) * yf;
    gradY = (h01 - h00) * (1 - xf) + (h11 - h10) * xf;
}

void Erosion::applyHydraulicErosion(
    std::vector<float> &heightmap,
    const ErosionParams &p)
{
    const int size = p.mapSize;

    // Precompute erosion brush (offsets + weights) for given radius
    int r = p.brushRadius;
    std::vector<int> brushIndices;
    std::vector<float> brushWeights;
    brushIndices.reserve((2 * r + 1) * (2 * r + 1));
    brushWeights.reserve((2 * r + 1) * (2 * r + 1));

    for (int by = -r; by <= r; ++by)
    {
        for (int bx = -r; bx <= r; ++bx)
        {
            float sqr = float(bx * bx + by * by);
            if (sqr <= r * r)
            {
                brushIndices.push_back(by * size + bx);
                // weight falls off with distance (linear), center has highest weight
                float w = 1.0f - std::sqrt(sqr) / float(r);
                brushWeights.push_back(w);
            }
        }
    }

    // normalize weights
    float wsum = 0.0f;
    for (float w : brushWeights)
        wsum += w;
    if (wsum <= 0.0f)
        wsum = 1.0f;
    for (float &w : brushWeights)
        w /= wsum;

    // prepare random starting indices (shuffled)
    std::vector<int> rndIndices(size * size);
    for (int i = 0; i < size * size; ++i)
        rndIndices[i] = i;
    std::mt19937 rng(1337);
    std::shuffle(rndIndices.begin(), rndIndices.end(), rng);

    const int border = p.borderSize;

    for (int drop = 0; drop < p.numDrops; ++drop)
    {
        int startIdx = rndIndices[drop % (size * size)];
        float posX = float(startIdx % size);
        float posY = float(startIdx / size);

        float dirX = 0.0f, dirY = 0.0f;
        float speed = p.startSpeed;
        float water = p.startWater;
        float sediment = 0.0f;

        for (int lifetime = 0; lifetime < p.maxLifetime; ++lifetime)
        {
            int nodeX = int(posX);
            int nodeY = int(posY);
            float cellOffsetX = posX - nodeX;
            float cellOffsetY = posY - nodeY;

            // compute height and gradient at current pos
            float height, gradX, gradY;
            sampleHeightAndGradient(heightmap, size, posX, posY, height, gradX, gradY);

            // update direction and move
            dirX = dirX * p.inertia - gradX * (1.0f - p.inertia);
            dirY = dirY * p.inertia - gradY * (1.0f - p.inertia);

            float len = std::sqrt(dirX * dirX + dirY * dirY);
            if (len < 0.01f)
            {
                // small jitter
                std::uniform_real_distribution<float> jitter(-1.0f, 1.0f);
                dirX = jitter(rng);
                dirY = jitter(rng);
                len = std::sqrt(dirX * dirX + dirY * dirY);
            }
            dirX /= len;
            dirY /= len;

            posX += dirX;
            posY += dirY;

            // stop if out of bounds
            if (posX < border || posX >= size - border || posY < border || posY >= size - border)
                break;

            // new height at moved position
            float newH, ngX, ngY;
            sampleHeightAndGradient(heightmap, size, posX, posY, newH, ngX, ngY);
            float deltaHeight = newH - height;

            float sedimentCapacity = std::max(-deltaHeight * speed * water * p.sedimentCapacityFactor, p.minSedimentCapacity);

            int dropletNodeX = int(posX);
            int dropletNodeY = int(posY);
            int dropletIndex = dropletNodeY * size + dropletNodeX;

            if (sediment > sedimentCapacity || deltaHeight > 0.0f)
            {
                // deposit
                float amountToDeposit = (deltaHeight > 0.0f) ? std::min(deltaHeight, sediment) : (sediment - sedimentCapacity) * p.depositSpeed;
                amountToDeposit = std::max(0.0f, amountToDeposit);
                sediment -= amountToDeposit;

                // Deposit locally to the four surrounding nodes (bilinear) this
                // allows filling small pits and encourages channel formation
                int xi = dropletNodeX;
                int yi = dropletNodeY;
                heightmap[yi * size + xi] += amountToDeposit * (1 - cellOffsetX) * (1 - cellOffsetY);
                heightmap[yi * size + (xi + 1)] += amountToDeposit * cellOffsetX * (1 - cellOffsetY);
                heightmap[(yi + 1) * size + xi] += amountToDeposit * (1 - cellOffsetX) * cellOffsetY;
                heightmap[(yi + 1) * size + (xi + 1)] += amountToDeposit * cellOffsetX * cellOffsetY;
            }
            else
            {
                // erode using brush
                float amountToErode = std::min((sedimentCapacity - sediment) * p.erodeSpeed, -deltaHeight);
                if (amountToErode > 0.0f)
                {
                    for (size_t bi = 0; bi < brushIndices.size(); ++bi)
                    {
                        int offset = brushIndices[bi];
                        int target = dropletIndex + offset;
                        // clamp to map
                        int tx = dropletNodeX + (offset % size);
                        int ty = dropletNodeY + (offset / size);
                        if (tx < 0 || tx >= size || ty < 0 || ty >= size)
                            continue;
                        float weighted = amountToErode * brushWeights[bi];
                        float deltaSed = std::min(heightmap[target], weighted);
                        heightmap[target] -= deltaSed;
                        sediment += deltaSed;
                    }
                }
            }

            // update speed & water (use compute-shader style)
            speed = std::sqrt(std::max(0.0f, speed * speed + deltaHeight * p.gravity));
            water *= (1.0f - p.evaporateSpeed);
            if (water <= 0.0f)
                break;
        }
    }

    // clamp
    for (float &h : heightmap)
        h = clampf(h, 0.0f, 1.0f);

    // smoothing passes
    if (p.smoothingPasses > 0)
    {
        std::vector<float> src = heightmap;
        std::vector<float> dst = heightmap;
        for (int pass = 0; pass < p.smoothingPasses; ++pass)
        {
            for (int y = 1; y < size - 1; ++y)
            {
                for (int x = 1; x < size - 1; ++x)
                {
                    int i = y * size + x;
                    float s = 0.0f;
                    s += src[i] * 4.0f;
                    s += src[(y - 1) * size + (x - 1)];
                    s += src[(y - 1) * size + x];
                    s += src[(y - 1) * size + (x + 1)];
                    s += src[y * size + (x - 1)];
                    s += src[y * size + (x + 1)];
                    s += src[(y + 1) * size + (x - 1)];
                    s += src[(y + 1) * size + x];
                    s += src[(y + 1) * size + (x + 1)];
                    dst[i] = s / 12.0f;
                }
            }
            src.swap(dst);
        }
        if (p.smoothingPasses % 2 == 1)
            heightmap = src;
        else
            heightmap = dst;
    }
}
