#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "deps/imgui/imgui.h"
#include "deps/imgui/backends/imgui_impl_glfw.h"
#include "deps/imgui/backends/imgui_impl_opengl3.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include "CPU/Erosion.h"
#include "deps/FastNoiseLite.h"
#include "helper.h"

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
    // forward to ImGui
    ImGui_ImplGlfw_MouseButtonCallback(w, button, action, mods);
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        if (action == GLFW_PRESS)
            updateCursorState(w);
        return;
    }
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
    // forward to ImGui
    ImGui_ImplGlfw_CursorPosCallback(w, xpos, ypos);
    ImGuiIO &io = ImGui::GetIO();
    double dx = xpos - lastX;
    double dy = ypos - lastY;
    if (io.WantCaptureMouse)
    {
        // update cursor state to avoid jumps when leaving UI
        lastX = xpos;
        lastY = ypos;
        return;
    }
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
    ImGui_ImplGlfw_ScrollCallback(w, xoffset, yoffset);
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;
    if (yoffset == 0.0)
        return;
    float zoomFactor = (yoffset > 0) ? 0.85f : 1.15f;
    cam.distance *= zoomFactor;
    if (cam.distance < 0.05f)
        cam.distance = 0.05f;
}

static void key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    ImGui_ImplGlfw_KeyCallback(w, key, scancode, action, mods);
    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
        return;
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

float fbm(FastNoiseLite &noise, float x, float y, const PerlinParams &p, const std::vector<Vector2> &offsets)
{
    float frequency = 1.0f / p.initialScale;
    float amplitude = 1.0f;
    float value = 0.0f;
    float ampSum = 0.0f;

    for (int o = 0; o < p.numOctaves; o++)
    {
        float nx = offsets[o].x + x * frequency;
        float ny = offsets[o].y + y * frequency;

        float n = noise.GetNoise(nx, ny); // [-1,1]

        value += n * amplitude;
        ampSum += amplitude;

        amplitude *= p.persistence;
        frequency *= p.lacunarity;
    }

    return value / ampSum; // normalized to [-1,1]
}

static std::vector<float> generateHeightMap(int size, const PerlinParams &p)
{
    std::vector<float> map(size * size);

    std::mt19937 rgen(70);
    std::uniform_real_distribution<float> dist(-10000.f, 10000.f);

    std::vector<Vector2> offsets(p.numOctaves);
    for (auto &v : offsets)
        v = {dist(rgen), dist(rgen)};

    FastNoiseLite noise;
    noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    noise.SetFrequency(1.0f);

    auto fbm = [&](float x, float y)
    {
        float frequency = 1.0f / p.initialScale;
        float amplitude = 1.0f;
        float value = 0.0f;
        float ampSum = 0.0f;

        for (int o = 0; o < p.numOctaves; o++)
        {
            float nx = offsets[o].x + x * frequency;
            float ny = offsets[o].y + y * frequency;

            float n = noise.GetNoise(nx, ny);

            value += n * amplitude;
            ampSum += amplitude;

            amplitude *= p.persistence;
            frequency *= p.lacunarity;
        }

        return value / ampSum;
    };

    auto ridged = [&](float n)
    {
        n = std::abs(n);
        n = 1.0f - n;
        return n * n;
    };

    auto ridgedFBM = [&](float x, float y)
    {
        float frequency = 1.0f / p.initialScale;
        float amplitude = 0.5f;
        float value = 0.0f;
        float weight = 1.0f;

        for (int o = 0; o < p.numOctaves; o++)
        {
            float nx = offsets[o].x + x * frequency;
            float ny = offsets[o].y + y * frequency;

            float n = noise.GetNoise(nx, ny);
            float r = ridged(n);

            r *= weight;
            weight = std::clamp(r * 2.0f, 0.0f, 1.0f);

            value += r * amplitude;

            frequency *= p.lacunarity;
            amplitude *= p.persistence;
        }

        return value;
    };

    auto erosionBias = [&](float h)
    {
        h = (h + 1.0f) * 0.5f; // [-1,1] to [0,1]
        return std::pow(h, 1.4f);
    };

    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float base = fbm((float)x, (float)y);
            float ridges = ridgedFBM((float)x, (float)y);

            float h = base * 0.5f + ridges * 0.5f;
            h = erosionBias(h);

            map[y * size + x] = h;
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

    // ----- Setup Dear ImGui -----
    const char *glsl_version = "#version 330";
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    // Do not let ImGui install GLFW callbacks; we install our own and forward events.
    ImGui_ImplGlfw_InitForOpenGL(win, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    MapParams mparams;
    ErosionParams eparams;
    PerlinParams pparams;
    std::vector<float> heightmap = generateHeightMap(mparams.size, pparams);

    // apply hydraulic erosion
    // Erosion::applyHydraulicErosion(heightmap, eparams);

    // build mesh from heightmap
    Mesh mesh = buildGrid(mparams.size, heightmap);

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
    // Mat4 proj = perspective(45.0f * 3.14159265f / 180.0f, 1280.0f / 720.0f, 0.1f, 100.0f);
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

        float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
        Mat4 proj = perspective(45.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);

        // compute camera eye from spherical coords
        float eyeX = cam.targetX + cam.distance * std::cos(cam.pitch) * std::sin(cam.yaw);
        float eyeY = cam.targetY + cam.distance * std::sin(cam.pitch);
        float eyeZ = cam.targetZ + cam.distance * std::cos(cam.pitch) * std::cos(cam.yaw);
        Mat4 view = lookAt(eyeX, eyeY, eyeZ, cam.targetX, cam.targetY, cam.targetZ, 0.0f, 1.0f, 0.0f);
        Mat4 mvp = mul(proj, mul(view, model));

        glUseProgram(prog);

        glUniform3f(glGetUniformLocation(prog, "uBaseColour"), 0.2f, 0.8f, 0.3f);

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

        // ----- ImGui UI -----
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Erosion Controls");
        ImGui::Text("Map / Erosion Parameters");
        if (ImGui::SliderInt("Map Size", &mparams.size, 32, 1024))
        {
            // clamp to reasonable sizes
            if (mparams.size < 32)
                mparams.size = 32;
        }
        ImGui::SliderInt("Num Drops", &eparams.numDrops, 0, 1000000);
        ImGui::SliderInt("Max Lifetime", &eparams.maxLifetime, 1, 1000);
        ImGui::SliderFloat("Inertia", &eparams.inertia, 0.0f, 1.0f);
        ImGui::SliderFloat("Sediment Capacity Factor", &eparams.sedimentCapacityFactor, 0.0f, 10.0f);
        ImGui::SliderFloat("Min Sediment Capacity", &eparams.minSedimentCapacity, 0.0f, 1.0f);
        ImGui::SliderFloat("Erode Speed", &eparams.erodeSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Deposit Speed", &eparams.depositSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Evaporate Speed", &eparams.evaporateSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Gravity", &eparams.gravity, 0.0f, 20.0f);

        ImGui::Separator();
        ImGui::Text("Perlin Noise");
        ImGui::SliderInt("Octaves", &pparams.numOctaves, 1, 12);
        ImGui::SliderFloat("Persistence", &pparams.persistence, 0.0f, 1.0f);
        ImGui::SliderFloat("Lacunarity", &pparams.lacunarity, 1.0f, 6.0f);
        ImGui::SliderFloat("Initial Scale", &pparams.initialScale, 1.0f, 2000.0f);

        if (ImGui::Button("Regenerate Heightmap"))
        {
            heightmap = generateHeightMap(mparams.size, pparams);
            mesh = buildGrid(mparams.size, heightmap);
            // upload new buffers
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, nbo);
            glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(float), mesh.normals.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply Erosion"))
        {
            Erosion::applyHydraulicErosion(heightmap, eparams, mparams.size);
            mesh = buildGrid(mparams.size, heightmap);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, nbo);
            glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(float), mesh.normals.data(), GL_STATIC_DRAW);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);
        }

        ImGui::Separator();
        ImGui::Text("Mesh info: %d vertices, %d indices", (int)(mesh.vertices.size() / 3), (int)mesh.indices.size());
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    // Shutdown ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
