cmake_minimum_required(VERSION 3.16)
project(ErosionSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include(FetchContent)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable building documentation" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable building examples" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Disable building tests" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# GLAD
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v0.1.35
)
FetchContent_MakeAvailable(glad)

# OpenGL
find_package(OpenGL REQUIRED)

# ImGui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/imgui)

# Executable
add_executable(ErosionSim
    src/main.cpp
    src/helper.cpp
    src/CPU/erosion.cpp
    src/CPU/map.cpp
    src/GPU0/map.cu
    src/GPU0/erosion.cu
    src/GPU1/map.cu
    src/GPU1/erosion.cu
    src/GPU2/map.cu
    src/GPU2/erosion.cu
    src/GPU3/map.cu
    src/GPU3/erosion.cu
    src/GPUgridBased/map.cu
    src/GPUgridBased/erosion.cu
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(ErosionSim PRIVATE
    ${glfw_SOURCE_DIR}/include
    ${glad_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_link_libraries(ErosionSim PRIVATE glfw glad OpenGL::GL)

if(UNIX)
    find_package(X11 REQUIRED)
    target_link_libraries(ErosionSim PRIVATE ${X11_LIBRARIES})
endif()

target_compile_definitions(ErosionSim PRIVATE "GLFW_INCLUDE_NONE")

configure_file(shaders/terrain.vert shaders/terrain.vert COPYONLY)
configure_file(shaders/terrain.frag shaders/terrain.frag COPYONLY)
