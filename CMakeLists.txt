cmake_minimum_required(VERSION 3.16)
project(ErosionSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "Disable building documentation" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Disable building examples" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "Disable building tests" FORCE)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG        v0.1.35
)
FetchContent_MakeAvailable(glad)

find_package(OpenGL REQUIRED)

add_executable(ErosionSim
  src/main.cpp
  src/Erosion.cpp
  src/Mesh.cpp
)

target_include_directories(ErosionSim PRIVATE ${glfw_SOURCE_DIR}/include ${glad_SOURCE_DIR}/include src)
target_link_libraries(ErosionSim PRIVATE glfw glad OpenGL::GL)

if (UNIX)
  find_package(X11 REQUIRED)
  target_link_libraries(ErosionSim PRIVATE ${X11_LIBRARIES})
endif()

target_compile_definitions(ErosionSim PRIVATE "GLFW_INCLUDE_NONE")

configure_file(shaders/terrain.vert shaders/terrain.vert COPYONLY)
configure_file(shaders/terrain.frag shaders/terrain.frag COPYONLY)
