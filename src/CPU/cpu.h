#pragma once
#include <vector>
#include <cstdint>
#include "helper.h"

class Cpu
{
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize);
    static std::vector<float> generateHeightMap(int size, const PerlinParams &p);
};
