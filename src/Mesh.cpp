#include "Mesh.h"
#include <cmath>
#include <algorithm>
#include <vector>

static inline int idx(int x, int y, int size) { return y * size + x; }

float scale = 20.0f;
float elevationScale = 10.0f;

Mesh MeshBuilder::buildGrid(int size, const std::vector<float> &heightmap){
    Mesh m;
    m.vertices.reserve(size * size * 3);
    m.normals.resize(size * size * 3, 0.0f); // initialize to zero

    for(int y=0;y<size;y++){
        for(int x=0;x<size;x++){
            float nx = (float)x / (size - 1) - 0.5f;
            float ny = (float)y / (size - 1) - 0.5f;
            float z = heightmap[idx(x,y,size)];
            m.vertices.push_back(nx);
            m.vertices.push_back(z);
            m.vertices.push_back(ny);
        }
    }

    for(int y=0;y<size-1;y++){
        for(int x=0;x<size-1;x++){
            int i0 = idx(x,y,size);
            int i1 = idx(x+1,y,size);
            int i2 = idx(x,y+1,size);
            int i3 = idx(x+1,y+1,size);

            // first triangle
            m.indices.push_back(i0);
            m.indices.push_back(i2);
            m.indices.push_back(i1);

            // second triangle
            m.indices.push_back(i1);
            m.indices.push_back(i2);
            m.indices.push_back(i3);
        }
    }

    auto addNormal = [&](int i, float nx, float ny, float nz){
        m.normals[i*3 + 0] += nx;
        m.normals[i*3 + 1] += ny;
        m.normals[i*3 + 2] += nz;
    };

    for(size_t t=0; t < m.indices.size(); t += 3){
        int i0 = m.indices[t+0];
        int i1 = m.indices[t+1];
        int i2 = m.indices[t+2];

        // vertex positions
        float x0 = m.vertices[i0*3+0], y0 = m.vertices[i0*3+1], z0 = m.vertices[i0*3+2];
        float x1 = m.vertices[i1*3+0], y1 = m.vertices[i1*3+1], z1 = m.vertices[i1*3+2];
        float x2 = m.vertices[i2*3+0], y2 = m.vertices[i2*3+1], z2 = m.vertices[i2*3+2];

        // edges
        float ex1 = x1 - x0, ey1 = y1 - y0, ez1 = z1 - z0;
        float ex2 = x2 - x0, ey2 = y2 - y0, ez2 = z2 - z0;

        // cross product to get face normal
        float nx = ey1*ez2 - ez1*ey2;
        float ny = ez1*ex2 - ex1*ez2;
        float nz = ex1*ey2 - ey1*ex2;

        // add face normal to each vertex
        addNormal(i0, nx, ny, nz);
        addNormal(i1, nx, ny, nz);
        addNormal(i2, nx, ny, nz);
    }

    // normalize per-vertex normals
    for(int i=0; i<size*size; ++i){
        float nx = m.normals[i*3+0];
        float ny = m.normals[i*3+1];
        float nz = m.normals[i*3+2];
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if(len > 0.0f){
            m.normals[i*3+0] = nx / len;
            m.normals[i*3+1] = ny / len;
            m.normals[i*3+2] = nz / len;
        } else {
            m.normals[i*3+0] = 0.0f;
            m.normals[i*3+1] = 1.0f; // default up
            m.normals[i*3+2] = 0.0f;
        }
    }

    return m;
}