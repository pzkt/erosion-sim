#pragma once
#include <vector>
#include <cstdint>
#include "helper.h"

class Erosion
{
public:
    static void applyHydraulicErosion(std::vector<float> &heightmap, const ErosionParams &p, int mapSize);
};
