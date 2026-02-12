# ErosionSim

Running this project requires CMake, a C++17 toolchain, git + network access to fetch dependencies (glad and glfw), OpenGL system libraries, an NVIDIA GPU (tested and optimized for the NVIDIA GeForce RTX 3060 Ti), the CUDA toolkit + NVCC.

```bash
mkdir build
cmake -S . -B build && cmake --build build -j
./build/ErosionSim
```

Or use the dockerfile, but you'll still need the NVIDIA container toolkit and forward X11 from host:
```bash
docker build -t erosion-sim .
xhost +local:docker
docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix erosion-sim
```