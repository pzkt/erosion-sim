#pragma once
#include <vector>
#include <cstdint>
#include "helper.h"

class Map
{
public:
    static std::vector<float> generateHeightMap(int size, const PerlinParams &p);
};
