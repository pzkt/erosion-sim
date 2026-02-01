#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include "Erosion.h"
#include "Mesh.h"
#include "FastNoiseLite.h"
#include "MatrixHelper.h"

static double lastX = 0.0, lastY = 0.0;
static bool leftDown = false, rightDown = false;
static camParams cam;

static void updateCursorState(GLFWwindow *win)
{
    double cx, cy;
    glfwGetCursorPos(win, &cx, &cy);
    lastX = cx;
    lastY = cy;
}

// Input callbacks
static void mouse_button_cb(GLFWwindow *w, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        leftDown = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        rightDown = (action == GLFW_PRESS);
    }

    if (action == GLFW_PRESS)
        updateCursorState(w);
}

static void cursor_pos_cb(GLFWwindow *w, double xpos, double ypos)
{
    double dx = xpos - lastX;
    double dy = ypos - lastY;
    lastX = xpos;
    lastY = ypos;
    if (leftDown)
    {
        float sens = 0.005f;
        cam.yaw -= (float)dx * sens;
        cam.pitch += (float)dy * sens;
        const float limit = 1.49f;
        if (cam.pitch > limit)
            cam.pitch = limit;
        if (cam.pitch < -limit)
            cam.pitch = -limit;
    }
    else if (rightDown)
    {
        float camX = cam.targetX + cam.distance * std::cos(cam.pitch) * std::sin(cam.yaw);
        float camY = cam.targetY + cam.distance * std::sin(cam.pitch);
        float camZ = cam.targetZ + cam.distance * std::cos(cam.pitch) * std::cos(cam.yaw);
        // forward = normalize(target - eye)
        float fx = cam.targetX - camX, fy = cam.targetY - camY, fz = cam.targetZ - camZ;
        vnorm(fx, fy, fz);
        // right = normalize(cross(forward, up))
        float rx, ry, rz;
        vcross(fx, fy, fz, 0.0f, 1.0f, 0.0f, rx, ry, rz);
        vnorm(rx, ry, rz);
        float ux, uy, uz;
        vcross(rx, ry, rz, fx, fy, fz, ux, uy, uz);
        float panSpeed = 0.0015f * cam.distance;
        cam.targetX += -rx * (float)dx * panSpeed + ux * (float)dy * panSpeed;
        cam.targetY += -ry * (float)dx * panSpeed + uy * (float)dy * panSpeed;
        cam.targetZ += -rz * (float)dx * panSpeed + uz * (float)dy * panSpeed;
    }
}

static void scroll_cb(GLFWwindow *w, double xoffset, double yoffset)
{
    if (yoffset == 0.0)
        return;
    float zoomFactor = (yoffset > 0) ? 0.85f : 1.15f;
    cam.distance *= zoomFactor;
    if (cam.distance < 0.05f)
        cam.distance = 0.05f;
}

static void key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        cam = camParams();
        updateCursorState(w);
    }
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    int ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char buf[1024];
        glGetShaderInfoLog(s, 1024, nullptr, buf);
        std::cerr << "Shader compile error: " << buf << "\n";
    }
    return s;
}

static std::string loadFile(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return std::string();
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0)
    {
        fclose(f);
        return std::string();
    }
    std::string s;
    s.resize(sz);
    size_t read = fread(&s[0], 1, sz, f);
    fclose(f);
    if (read != (size_t)sz)
        s.resize(read);
    return s;
}

static std::vector<float> generateHightMap(int size, PerlinParams p)
{
    std::vector<float> map = std::vector<float>(size * size, 0.0f);
    std::mt19937 rgen(70);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<Vector2> offsets(p.numOctaves);

    for (auto &v : offsets)
    {
        v = Vector2{
            static_cast<float>(dist(rgen)),
            static_cast<float>(dist(rgen))};
    }

    float minValue = std::numeric_limits<float>::max();
    float maxValue = std::numeric_limits<float>::lowest();

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType::NoiseType_Perlin);

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float noiseValue = 0.0f;
            float scale = p.initialScale;
            float weight = 1.0f;

            for (int octave = 0; octave < p.numOctaves; octave++)
            {
                Vector2 point;
                point.x = offsets[octave].x + (float)x / (float)size * scale;
                point.y = offsets[octave].y + (float)y / (float)size * scale;

                float n = noise.GetNoise(point.x, point.y);
                n = (n + 1.0f) * 0.5f; // force [0,1]
                noiseValue += n * weight;

                weight *= p.persistence;
                scale *= p.lacunarity;
            }

            map[y * size + x] = noiseValue;
            minValue = (noiseValue < minValue) ? noiseValue : minValue;
            maxValue = (noiseValue > maxValue) ? noiseValue : maxValue;
        }
    }

    if (maxValue != minValue)
    {
        for (auto &v : map)
        {
            v = (v - minValue) / (maxValue - minValue);
        }
    }

    return map;
}

int main()
{
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *win = glfwCreateWindow(1280, 720, "ErosionSim", nullptr, nullptr);
    if (!win)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    ErosionParams eparams;
    PerlinParams pparams;
    std::vector<float> heightmap = generateHightMap(eparams.mapSize, pparams);

    // apply hydraulic erosion
    // Erosion::applyHydraulicErosion(heightmap, eparams);

    // build mesh from heightmap
    Mesh mesh = MeshBuilder::buildGrid(eparams.mapSize, heightmap);

    // upload to GPU
    GLuint vao, vbo, nbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &nbo);
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(float), mesh.normals.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);

    // load shaders from configured copy
    std::string vsrc = loadFile("shaders/terrain.vert");
    std::string fsrc = loadFile("shaders/terrain.frag");
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc.c_str());
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    // set attributes
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glEnable(GL_DEPTH_TEST);

    // basic camera/matrices
    Mat4 proj = perspective(45.0f * 3.14159265f / 180.0f, 1280.0f / 720.0f, 0.1f, 100.0f);
    Mat4 model = mul(translate(0.0f, -0.3f, 0.0f), scale(2.0f));

    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uModel = glGetUniformLocation(prog, "uModel");
    GLint uLightDirLoc = glGetUniformLocation(prog, "uLightDir");
    GLint uViewPosLoc = glGetUniformLocation(prog, "uViewPos");

    // set input callbacks
    glfwSetMouseButtonCallback(win, mouse_button_cb);
    glfwSetCursorPosCallback(win, cursor_pos_cb);
    glfwSetScrollCallback(win, scroll_cb);
    glfwSetKeyCallback(win, key_cb);
    updateCursorState(win);

    while (!glfwWindowShouldClose(win))
    {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.18f, 0.18f, 0.19f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // compute camera eye from spherical coords
        float eyeX = cam.targetX + cam.distance * std::cos(cam.pitch) * std::sin(cam.yaw);
        float eyeY = cam.targetY + cam.distance * std::sin(cam.pitch);
        float eyeZ = cam.targetZ + cam.distance * std::cos(cam.pitch) * std::cos(cam.yaw);
        Mat4 view = lookAt(eyeX, eyeY, eyeZ, cam.targetX, cam.targetY, cam.targetZ, 0.0f, 1.0f, 0.0f);
        Mat4 mvp = mul(proj, mul(view, model));

        glUseProgram(prog);

        glUniform3f(glGetUniformLocation(prog, "uBaseColour"), 0.2f, 0.8f, 0.3f);
        glUniform3f(glGetUniformLocation(prog, "uRimColour"), 1.0f, 1.0f, 1.0f);
        glUniform1f(glGetUniformLocation(prog, "uRimPower"), 2.0f);
        glUniform1f(glGetUniformLocation(prog, "uRimFac"), 0.5f);
        glUniform1f(glGetUniformLocation(prog, "uMinHeight"), 0.0f); // min Y of terrain
        glUniform1f(glGetUniformLocation(prog, "uMaxHeight"), 1.0f); // max Y of terrain

        glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_FALSE, model.m);
        glUniformMatrix4fv(glGetUniformLocation(prog, "uView"), 1, GL_FALSE, view.m);
        glUniformMatrix4fv(glGetUniformLocation(prog, "uProj"), 1, GL_FALSE, proj.m);

        // position
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

        // normal
        glBindBuffer(GL_ARRAY_BUFFER, nbo);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

        // glUniformMatrix4fv(uMVP, 1, GL_FALSE, mvp.m);
        // glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);

        if (uViewPosLoc >= 0)
        {
            glUniform3f(uViewPosLoc, eyeX, eyeY, eyeZ);
        }

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, (GLsizei)mesh.indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
