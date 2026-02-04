#pragma once
#include <cmath>
#include <vector>

struct Vector2
{
    float x;
    float y;
};

struct Mat4
{
    float m[16];
};

struct Mesh
{
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<unsigned int> indices;
};

struct camParams
{
    float yaw = -2.0f;
    float pitch = 1.0f;
    float distance = 45.0f;
    float targetX = 0.0f;
    float targetY = -0.1f;
    float targetZ = 0.0f;
};

struct MapParams
{
    int size = 2048; // 512
    float scale = 5.0f;
    float elevationScale = 8.0f;
};

struct ErosionParams
{
    int numDrops = 10000000;
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
    int numOctaves = 10;
    float persistence = 0.45f;
    float lacunarity = 2.1f;
    float initialScale = 400.0f;
};

enum class ComputeMode
{
    CPU,
    GPU0,
    GPU1,
    GPU2
};

static inline float
vlen(float x, float y, float z)
{
    return std::sqrt(x * x + y * y + z * z);
}

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

static inline int idx(int x, int y, int size) { return y * size + x; }

Mat4 perspective(float fovy, float aspect, float zn, float zf);
Mat4 translate(float x, float y, float z);
Mat4 scale(float s);
Mat4 mul(const Mat4 &a, const Mat4 &b);
void vcross(const float ax, const float ay, const float az, const float bx, const float by, const float bz, float &rx, float &ry, float &rz);
Mat4 lookAt(const float eyeX, const float eyeY, const float eyeZ, const float centerX, const float centerY, const float centerZ, const float upX, const float upY, const float upZ);
Mesh buildGrid(const std::vector<float> &heightmap, MapParams mparams);