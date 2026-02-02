#pragma once
#include <vector>
#include <cstdint>

struct ErosionParams
{
    int mapSize = 256;
    int numDrops = 150000;
    int maxLifetime = 40;
    float inertia = 0.3f;
    float sedimentCapacityFactor = 4.0f;
    float minSedimentCapacity = 0.01f;
    float erodeSpeed = 0.02f;
    float depositSpeed = 0.4f;
    float evaporateSpeed = 0.03f;
    float gravity = 3.0f;
    int smoothingPasses = 0;
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
