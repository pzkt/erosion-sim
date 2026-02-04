# ErosionSim

Running this project requires CMake, a C++17 toolchain, git + network access to fetch dependencies (glad and glfw), OpenGL system libraries, an NVIDIA GPU (tested and optimized for the NVIDIA GeForce RTX 3060 Ti), the CUDA toolkit + NVCC.
```bash
mkdir build
cmake -S . -B build && cmake --build build -j
./build/ErosionSim
```