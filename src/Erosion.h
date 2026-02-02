#pragma once
#include <vector>
#include <cstdint>

struct ErosionParams
{
    int mapSize = 512;
    int numDrops = 600000;
    int maxLifetime = 30;
    float inertia = 0.3f;
    float sedimentCapacityFactor = 4.0f;
    float minSedimentCapacity = 0.01f;
    float erodeSpeed = 0.3f;
    float depositSpeed = 0.3f;
    float evaporateSpeed = 0.01f;
    float gravity = 4.0f;
    int brushRadius = 3;
    float startSpeed = 1.0f;
    float startWater = 1.0f;
    int borderSize = 4;
};

struct PerlinParams
{
    int numOctaves = 6;
    float persistence = 0.45f;
    float lacunarity = 2.1f;
    float initialScale = 200.0f;
};

class Erosion
{
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p);
};
