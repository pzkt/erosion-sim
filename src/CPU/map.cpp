#include "map.h"
#include "deps/FastNoiseLite.h"
#include "helper.h"
#include <vector>
#include <random>
#include <algorithm>

float fbm(FastNoiseLite &noise, float x, float y, const PerlinParams &p, const std::vector<Vector2> &offsets)
{
    float frequency = 1.0f / p.initialScale;
    float amplitude = 1.0f;
    float value = 0.0f;
    float ampSum = 0.0f;

    for (int o = 0; o < p.numOctaves; o++)
    {
        float nx = offsets[o].x + x * frequency;
        float ny = offsets[o].y + y * frequency;

        float n = noise.GetNoise(nx, ny); // [-1,1]

        value += n * amplitude;
        ampSum += amplitude;

        amplitude *= p.persistence;
        frequency *= p.lacunarity;
    }

    return value / ampSum; // normalized to [-1,1]
}

std::vector<float> Map::generateHeightMap(int size, const PerlinParams &p)
{
    std::vector<float> map(size * size);

    std::mt19937 rgen(70);
    std::uniform_real_distribution<float> dist(-10000.f, 10000.f);

    std::vector<Vector2> offsets(p.numOctaves);
    for (auto &v : offsets)
        v = {dist(rgen), dist(rgen)};

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFrequency(1.0f);

    auto fbm = [&](float x, float y)
    {
        float frequency = 1.0f / p.initialScale;
        float amplitude = 1.0f;
        float value = 0.0f;
        float ampSum = 0.0f;

        for (int o = 0; o < p.numOctaves; o++)
        {
            float nx = offsets[o].x + x * frequency;
            float ny = offsets[o].y + y * frequency;

            float n = noise.GetNoise(nx, ny);

            value += n * amplitude;
            ampSum += amplitude;

            amplitude *= p.persistence;
            frequency *= p.lacunarity;
        }

        return value / ampSum;
    };

    auto ridged = [&](float n)
    {
        n = std::abs(n);
        n = 1.0f - n;
        return n * n;
    };

    auto ridgedFBM = [&](float x, float y)
    {
        float frequency = 1.0f / p.initialScale;
        float amplitude = 0.5f;
        float value = 0.0f;
        float weight = 1.0f;

        for (int o = 0; o < p.numOctaves; o++)
        {
            float nx = offsets[o].x + x * frequency;
            float ny = offsets[o].y + y * frequency;

            float n = noise.GetNoise(nx, ny);
            float r = ridged(n);

            r *= weight;
            weight = std::clamp(r * 2.0f, 0.0f, 1.0f);

            value += r * amplitude;

            frequency *= p.lacunarity;
            amplitude *= p.persistence;
        }

        return value;
    };

    auto erosionBias = [&](float h)
    {
        h = (h + 1.0f) * 0.5f; // [-1,1] to [0,1]
        return std::pow(h, 1.4f);
    };

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float base = fbm((float)x, (float)y);
            float ridges = ridgedFBM((float)x, (float)y);

            float h = base * 0.5f + ridges * 0.5f;
            h = erosionBias(h);

            map[y * size + x] = h;
        }
    }

    return map;
}