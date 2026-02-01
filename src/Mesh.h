#pragma once
#include <vector>

struct Mesh {
    std::vector<float> vertices; // x,y,z
    std::vector<float> normals;  // nx,ny,nz
    std::vector<unsigned int> indices;
};

class MeshBuilder {
public:
    static Mesh buildGrid(int size, const std::vector<float> &heightmap);
};
