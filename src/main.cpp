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
#include <chrono>
#include "CPU/cpu.h"
#include "GPU0/gpu.h"
#include "GPU1/gpu.h"
#include "helper.h"

static double lastX = 0.0, lastY = 0.0;
static bool leftDown = false, rightDown = false;
static camParams cam;
static ComputeMode computeMode = ComputeMode::GPU0;

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

static Mesh buildAndUploadGrid(GLuint vbo, GLuint nbo, GLuint ebo, const std::vector<float> &heightmap, const MapParams &mparams)
{
    Mesh mesh = buildGrid(heightmap, mparams);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size() * sizeof(float), mesh.vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, nbo);
    glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * sizeof(float), mesh.normals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);
    return mesh;
}

static void applyHydraulicErosion(std::vector<float> &heightmap, int size, ErosionParams e)
{
    switch (computeMode)
    {
    case ComputeMode::CPU:
        Cpu::applyHydraulicErosion(heightmap, e, size);
        break;
    case ComputeMode::GPU0:
        Gpu0::applyHydraulicErosion(heightmap, e, size);
        break;
    case ComputeMode::GPU1:
        Gpu1::applyHydraulicErosion(heightmap, e, size);
        break;
    }
}

static void generateHeightmap(std::vector<float> &heightmap, int size, PerlinParams p)
{
    switch (computeMode)
    {
    case ComputeMode::CPU:
        heightmap = Cpu::generateHeightMap(size, p);
        break;
    case ComputeMode::GPU0:
        heightmap.resize((size_t)size * (size_t)size);
        Gpu0::generateHeightmap(heightmap.data(), size, p);
        break;
    case ComputeMode::GPU1:
        heightmap.resize((size_t)size * (size_t)size);
        Gpu1::generateHeightmap(heightmap.data(), size, p);
        break;
    }
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

    ImGuiStyle &style = ImGui::GetStyle();
    ImVec4 *colors = style.Colors;

    // Base grays
    ImVec4 gray = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    ImVec4 grayHover = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    ImVec4 grayActive = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
    ImVec4 lightGray = ImVec4(0.35f, 0.35f, 0.37f, 1.00f);
    ImVec4 brightGray = ImVec4(0.48f, 0.48f, 0.55f, 1.00f);

    ImVec4 textColor = ImVec4(0.6f, 0.68f, 1.0f, 1.0f);

    // Window
    colors[ImGuiCol_TitleBg] = lightGray;
    colors[ImGuiCol_TitleBgActive] = lightGray;
    colors[ImGuiCol_TitleBgCollapsed] = lightGray;

    // Frame
    colors[ImGuiCol_FrameBg] = gray;
    colors[ImGuiCol_FrameBgHovered] = grayHover;
    colors[ImGuiCol_FrameBgActive] = grayActive;

    // Buttons
    colors[ImGuiCol_Button] = gray;
    colors[ImGuiCol_ButtonHovered] = grayHover;
    colors[ImGuiCol_ButtonActive] = brightGray;

    // Slider grabs and checkboxes
    colors[ImGuiCol_SliderGrab] = lightGray;
    colors[ImGuiCol_SliderGrabActive] = brightGray;
    colors[ImGuiCol_CheckMark] = brightGray;

    // Resize grip (bottom-right corner)
    colors[ImGuiCol_ResizeGrip] = gray;
    colors[ImGuiCol_ResizeGripHovered] = grayHover;
    colors[ImGuiCol_ResizeGripActive] = grayActive;

    // Scrollbars
    colors[ImGuiCol_ScrollbarBg] = gray;
    colors[ImGuiCol_ScrollbarGrab] = gray;
    colors[ImGuiCol_ScrollbarGrabHovered] = grayHover;
    colors[ImGuiCol_ScrollbarGrabActive] = grayActive;

    colors[ImGuiCol_Border] = gray;

    // Do not let ImGui install GLFW callbacks; we install our own and forward events.
    ImGui_ImplGlfw_InitForOpenGL(win, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    bool autoRegenerate = false;
    bool autoErode = false;

    std::chrono::duration<double, std::milli> regenDuration = std::chrono::duration<double, std::milli>(0.0);
    std::chrono::duration<double, std::milli> erosionDuration = std::chrono::duration<double, std::milli>(0.0);

    MapParams mparams;
    ErosionParams eparams;
    PerlinParams pparams;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> heightmap((size_t)mparams.size * (size_t)mparams.size);
    generateHeightmap(heightmap, mparams.size, pparams);

    regenDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);

    // upload to GPU
    GLuint vao, vbo, nbo, ebo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &nbo);
    glGenBuffers(1, &ebo);

    Mesh mesh = buildAndUploadGrid(vbo, nbo, ebo, heightmap, mparams);

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
    GLint uView = glGetUniformLocation(prog, "uView");
    GLint uProj = glGetUniformLocation(prog, "uProj");
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

        glUniformMatrix4fv(uModel, 1, GL_FALSE, model.m);
        glUniformMatrix4fv(uView, 1, GL_FALSE, view.m);
        glUniformMatrix4fv(uProj, 1, GL_FALSE, proj.m);

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
        ImGui::Begin("Control Panel");

        ImGui::Text("Compute Mode");
        ImGui::Spacing();

        if (ImGui::RadioButton("CPU", computeMode == ComputeMode::CPU))
            computeMode = ComputeMode::CPU;
        ImGui::SameLine();
        if (ImGui::RadioButton("GPU (naïve)", computeMode == ComputeMode::GPU0))
            computeMode = ComputeMode::GPU0;
        ImGui::SameLine();
        if (ImGui::RadioButton("GPU (optimized)", computeMode == ComputeMode::GPU1))
            computeMode = ComputeMode::GPU1;

        ImGui::Separator();
        if (ImGui::BeginTable("Actions", 2, ImGuiTableFlags_SizingStretchSame))
        {

            ImGui::TableNextColumn();
            ImGui::Text("Heightmap");
            ImGui::Spacing();
            ImGui::Checkbox("Continuously Generate", &autoRegenerate);
            if (ImGui::Button("Regenerate Heightmap") || autoRegenerate)
            {
                auto start = std::chrono::high_resolution_clock::now();

                generateHeightmap(heightmap, mparams.size, pparams);
                mesh = buildAndUploadGrid(vbo, nbo, ebo, heightmap, mparams);

                regenDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
            }

            ImGui::Text("Regen Runtime:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, textColor);
            ImGui::Text("%d ms", static_cast<int>(regenDuration.count()));
            ImGui::PopStyleColor();

            ImGui::TableNextColumn();
            ImGui::Text("Erosion");
            ImGui::Spacing();
            ImGui::Checkbox("Continuously Erode", &autoErode);
            if (ImGui::Button("Apply Erosion") || autoErode)
            {
                auto start = std::chrono::high_resolution_clock::now();

                applyHydraulicErosion(heightmap, mparams.size, eparams);
                mesh = buildAndUploadGrid(vbo, nbo, ebo, heightmap, mparams);

                erosionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
            }

            ImGui::Text("Erode Runtime:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, textColor);
            if (static_cast<int>(erosionDuration.count()) == 0)
                ImGui::TextUnformatted("n/a");
            else
                ImGui::Text("%d ms", static_cast<int>(erosionDuration.count()));
            ImGui::PopStyleColor();
            ImGui::EndTable();
        }

        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Map Parameters");
        ImGui::SliderInt("Map Size", &mparams.size, 32, 2048);
        ImGui::SliderFloat("Scale", &mparams.scale, 1.0f, 10.0f);
        ImGui::SliderFloat("Elevation Scale", &mparams.elevationScale, 0.1f, 50.0f);
        ImGui::Separator();

        ImGui::Spacing();
        ImGui::Text("Perlin Noise");
        ImGui::SliderInt("Octaves", &pparams.numOctaves, 1, 12);
        ImGui::SliderFloat("Persistence", &pparams.persistence, 0.0f, 1.0f);
        ImGui::SliderFloat("Lacunarity", &pparams.lacunarity, 1.0f, 6.0f);
        ImGui::SliderFloat("Initial Scale", &pparams.initialScale, 1.0f, 2000.0f);
        ImGui::Separator();

        ImGui::Spacing();
        ImGui::Text("Erosion Parameters");
        ImGui::SliderInt("Num Drops", &eparams.numDrops, 0, 10000000);
        ImGui::SliderInt("Max Lifetime", &eparams.maxLifetime, 1, 1000);
        ImGui::SliderFloat("Inertia", &eparams.inertia, 0.0f, 1.0f);
        ImGui::SliderFloat("Sediment Capacity Factor", &eparams.sedimentCapacityFactor, 0.0f, 10.0f);
        ImGui::SliderFloat("Min Sediment Capacity", &eparams.minSedimentCapacity, 0.0f, 1.0f);
        ImGui::SliderFloat("Erode Speed", &eparams.erodeSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Deposit Speed", &eparams.depositSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Evaporate Speed", &eparams.evaporateSpeed, 0.0f, 1.0f);
        ImGui::SliderFloat("Gravity", &eparams.gravity, 0.0f, 20.0f);
        ImGui::Separator();

        ImGui::Text("Mesh Info: Vertices:");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, textColor);
        ImGui::Text("%d", (int)(mesh.vertices.size() / 3));
        ImGui::SameLine();
        ImGui::PopStyleColor();
        ImGui::Text("Indices:");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, textColor);
        ImGui::Text("%d", (int)mesh.indices.size());
        ImGui::PopStyleColor();
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
