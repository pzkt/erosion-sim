#pragma once
#include <cmath>

struct Mat4
{
    float m[16];
};

struct Vector2
{
    float x;
    float y;
};

struct camParams
{
    float yaw = -2.0f;
    float pitch = 1.0f;
    float distance = 5.0f;
    float targetX = 0.0f;
    float targetY = -0.1f;
    float targetZ = 0.0f;
};

struct MapParams
{
    int size = 512;
    float scale = 20.0f;
    float elevationScale = 10.0f;
};

struct ErosionParams
{
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
    float initialScale = 400.0f;
};

static inline float vlen(float x, float y, float z) { return std::sqrt(x * x + y * y + z * z); }
static inline void vnorm(float &x, float &y, float &z)
{
    float l = vlen(x, y, z);
    if (l > 0.0f)
    {
        x /= l;
        y /= l;
        z /= l;
    }
}

// Simple matrix helpers
static Mat4 perspective(float fovy, float aspect, float zn, float zf)
{
    float f = 1.0f / tanf(fovy * 0.5f);
    Mat4 M = {};
    M.m[0] = f / aspect;
    M.m[5] = f;
    M.m[10] = (zf + zn) / (zn - zf);
    M.m[11] = -1;
    M.m[14] = (2 * zf * zn) / (zn - zf);
    return M;
}

static Mat4 translate(float x, float y, float z)
{
    Mat4 M = {};
    M.m[0] = 1;
    M.m[5] = 1;
    M.m[10] = 1;
    M.m[15] = 1;
    M.m[12] = x;
    M.m[13] = y;
    M.m[14] = z;
    return M;
}
static Mat4 scale(float s)
{
    Mat4 M = {};
    M.m[0] = s;
    M.m[5] = s;
    M.m[10] = s;
    M.m[15] = 1;
    return M;
}
static Mat4 mul(const Mat4 &a, const Mat4 &b)
{
    Mat4 r = {};
    // column-major storage: element (row r, col c) is at index c*4 + r
    for (int c = 0; c < 4; c++)
    {
        for (int r0 = 0; r0 < 4; r0++)
        {
            float s = 0.0f;
            for (int k = 0; k < 4; k++)
            {
                s += a.m[k * 4 + r0] * b.m[c * 4 + k];
            }
            r.m[c * 4 + r0] = s;
        }
    }
    return r;
}

static inline void vcross(const float ax, const float ay, const float az, const float bx, const float by, const float bz, float &rx, float &ry, float &rz)
{
    rx = ay * bz - az * by;
    ry = az * bx - ax * bz;
    rz = ax * by - ay * bx;
}

static Mat4 lookAt(const float eyeX, const float eyeY, const float eyeZ, const float centerX, const float centerY, const float centerZ, const float upX, const float upY, const float upZ)
{
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    vnorm(fx, fy, fz);
    float sx, sy, sz;
    vcross(fx, fy, fz, upX, upY, upZ, sx, sy, sz);
    vnorm(sx, sy, sz);
    float ux, uy, uz;
    vcross(sx, sy, sz, fx, fy, fz, ux, uy, uz);
    Mat4 M = {};
    // column-major
    M.m[0] = sx;
    M.m[1] = ux;
    M.m[2] = -fx;
    M.m[3] = 0.0f;
    M.m[4] = sy;
    M.m[5] = uy;
    M.m[6] = -fy;
    M.m[7] = 0.0f;
    M.m[8] = sz;
    M.m[9] = uz;
    M.m[10] = -fz;
    M.m[11] = 0.0f;
    M.m[12] = -(sx * eyeX + sy * eyeY + sz * eyeZ);
    M.m[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
    M.m[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
    M.m[15] = 1.0f;
    return M;
}

static inline int idx(int x, int y, int size) { return y * size + x; }

struct Mesh
{
    std::vector<float> vertices; // x,y,z
    std::vector<float> normals;  // nx,ny,nz
    std::vector<unsigned int> indices;
};

static Mesh buildGrid(int size, const std::vector<float> &heightmap)
{
    float scale = 20.0f;
    float elevationScale = 10.0f;

    Mesh m;
    m.vertices.reserve(size * size * 3);
    m.normals.resize(size * size * 3, 0.0f);

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float nx = (float)x / (size - 1) - 0.5f;
            float ny = (float)y / (size - 1) - 0.5f;

            float z = heightmap[idx(x, y, size)];

            m.vertices.push_back(nx);
            m.vertices.push_back(z);
            m.vertices.push_back(ny);
        }
    }

    for (int y = 0; y < size - 1; y++)
    {
        for (int x = 0; x < size - 1; x++)
        {
            int i0 = idx(x, y, size);
            int i1 = idx(x + 1, y, size);
            int i2 = idx(x, y + 1, size);
            int i3 = idx(x + 1, y + 1, size);

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

    auto addNormal = [&](int i, float nx, float ny, float nz)
    {
        m.normals[i * 3 + 0] += nx;
        m.normals[i * 3 + 1] += ny;
        m.normals[i * 3 + 2] += nz;
    };

    for (size_t t = 0; t < m.indices.size(); t += 3)
    {
        int i0 = m.indices[t + 0];
        int i1 = m.indices[t + 1];
        int i2 = m.indices[t + 2];

        // vertex positions
        float x0 = m.vertices[i0 * 3 + 0], y0 = m.vertices[i0 * 3 + 1], z0 = m.vertices[i0 * 3 + 2];
        float x1 = m.vertices[i1 * 3 + 0], y1 = m.vertices[i1 * 3 + 1], z1 = m.vertices[i1 * 3 + 2];
        float x2 = m.vertices[i2 * 3 + 0], y2 = m.vertices[i2 * 3 + 1], z2 = m.vertices[i2 * 3 + 2];

        // edges
        float ex1 = x1 - x0, ey1 = y1 - y0, ez1 = z1 - z0;
        float ex2 = x2 - x0, ey2 = y2 - y0, ez2 = z2 - z0;

        // cross product to get face normal
        float nx = ey1 * ez2 - ez1 * ey2;
        float ny = ez1 * ex2 - ex1 * ez2;
        float nz = ex1 * ey2 - ey1 * ex2;

        // add face normal to each vertex
        addNormal(i0, nx, ny, nz);
        addNormal(i1, nx, ny, nz);
        addNormal(i2, nx, ny, nz);
    }

    // normalize per-vertex normals
    for (int i = 0; i < size * size; ++i)
    {
        float nx = m.normals[i * 3 + 0];
        float ny = m.normals[i * 3 + 1];
        float nz = m.normals[i * 3 + 2];
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 0.0f)
        {
            m.normals[i * 3 + 0] = nx / len;
            m.normals[i * 3 + 1] = ny / len;
            m.normals[i * 3 + 2] = nz / len;
        }
        else
        {
            m.normals[i * 3 + 0] = 0.0f;
            m.normals[i * 3 + 1] = 1.0f; // default up
            m.normals[i * 3 + 2] = 0.0f;
        }
    }

    return m;
}