#pragma once
#include <vector>
#include <cstdint>

struct ErosionParams {
    int mapSize = 256;
    int numDrops = 50000;
    int maxLifetime = 30;
    float inertia = 0.05f;
    float sedimentCapacityFactor = 4.0f;
    float minSedimentCapacity = 0.01f;
    float depositSpeed = 0.3f;
    float erodeSpeed = 0.3f;
    float evaporateSpeed = 0.01f;
    float gravity = 4.0f;
};

struct PerlinParams {
    int numOctaves = 7;
    float persistence = 0.4f;
    float lacunarity = 2.0f;
    float initialScale = 20.0f;
};

class Erosion {
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p);
};
