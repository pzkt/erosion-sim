#include "helper.h"
#include <cmath>
#include <vector>

// Simple matrix helpers
Mat4 perspective(float fovy, float aspect, float zn, float zf)
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

Mat4 translate(float x, float y, float z)
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

Mat4 scale(float s)
{
    Mat4 M = {};
    M.m[0] = s;
    M.m[5] = s;
    M.m[10] = s;
    M.m[15] = 1;
    return M;
}

Mat4 mul(const Mat4 &a, const Mat4 &b)
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

inline void vcross(const float ax, const float ay, const float az, const float bx, const float by, const float bz, float &rx, float &ry, float &rz)
{
    rx = ay * bz - az * by;
    ry = az * bx - ax * bz;
    rz = ax * by - ay * bx;
}

Mat4 lookAt(const float eyeX, const float eyeY, const float eyeZ, const float centerX, const float centerY, const float centerZ, const float upX, const float upY, const float upZ)
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

// turn heightmap into a grid mesh
Mesh buildGrid(const std::vector<float> &heightmap, MapParams mparams)
{
    int size = mparams.size;
    float scale = mparams.scale;
    float elevationScale = mparams.elevationScale;

    Mesh m;
    m.vertices.resize(size * size * 3);
    m.normals.assign(size * size * 3, 0.0f);
    m.indices.resize((size - 1) * (size - 1) * 6);

    int mapSizeWithBorder = (int)std::sqrt((float)heightmap.size());
    int border = 0;
    if (mapSizeWithBorder * mapSizeWithBorder == (int)heightmap.size() && mapSizeWithBorder >= size)
    {
        border = (mapSizeWithBorder - size) / 2;
    }

    // fill vertices
    for (int i = 0; i < size * size; ++i)
    {
        int x = i % size;
        int y = i / size;
        int meshMapIndex = y * size + x;

        float percentX = (float)x / (float)(size - 1);
        float percentY = (float)y / (float)(size - 1);
        float px = (percentX * 2.0f - 1.0f) * scale;
        float pz = (percentY * 2.0f - 1.0f) * scale;

        int hmIndex = meshMapIndex;
        if (border > 0)
            hmIndex = (y + border) * mapSizeWithBorder + x + border;

        float normalizedHeight = 0.0f;
        if (hmIndex >= 0 && hmIndex < (int)heightmap.size())
            normalizedHeight = heightmap[hmIndex];

        float py = (normalizedHeight - 0.5f) * elevationScale;

        m.vertices[meshMapIndex * 3 + 0] = px;
        m.vertices[meshMapIndex * 3 + 1] = py;
        m.vertices[meshMapIndex * 3 + 2] = pz;

        if (x != size - 1 && y != size - 1)
        {
            int t = (y * (size - 1) + x) * 6;
            m.indices[t + 0] = meshMapIndex + size;
            m.indices[t + 1] = meshMapIndex + size + 1;
            m.indices[t + 2] = meshMapIndex;

            m.indices[t + 3] = meshMapIndex + size + 1;
            m.indices[t + 4] = meshMapIndex + 1;
            m.indices[t + 5] = meshMapIndex;
        }
    }

    auto addNormal = [&](int vi, float nx, float ny, float nz)
    {
        m.normals[vi * 3 + 0] += nx;
        m.normals[vi * 3 + 1] += ny;
        m.normals[vi * 3 + 2] += nz;
    };

    for (size_t t = 0; t < m.indices.size(); t += 3)
    {
        int i0 = m.indices[t + 0];
        int i1 = m.indices[t + 1];
        int i2 = m.indices[t + 2];

        float x0 = m.vertices[i0 * 3 + 0], y0 = m.vertices[i0 * 3 + 1], z0 = m.vertices[i0 * 3 + 2];
        float x1 = m.vertices[i1 * 3 + 0], y1 = m.vertices[i1 * 3 + 1], z1 = m.vertices[i1 * 3 + 2];
        float x2 = m.vertices[i2 * 3 + 0], y2 = m.vertices[i2 * 3 + 1], z2 = m.vertices[i2 * 3 + 2];

        float ex1 = x1 - x0, ey1 = y1 - y0, ez1 = z1 - z0;
        float ex2 = x2 - x0, ey2 = y2 - y0, ez2 = z2 - z0;

        float nx = ey1 * ez2 - ez1 * ey2;
        float ny = ez1 * ex2 - ex1 * ez2;
        float nz = ex1 * ey2 - ey1 * ex2;

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
            m.normals[i * 3 + 1] = 1.0f;
            m.normals[i * 3 + 2] = 0.0f;
        }
    }

    return m;
}