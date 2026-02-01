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