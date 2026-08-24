cmake_minimum_required(VERSION 3.24)
project(CHIMERA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

if(NOT CHIMERA_TOOLCHAIN_ROOT)
    set(CHIMERA_TOOLCHAIN_ROOT "")
endif()
if(NOT CHIMERA_TOOLCHAIN_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
    set(CHIMERA_TOOLCHAIN_ROOT "$ENV{CONDA_PREFIX}")
endif()

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "CUDA toolkit root" FORCE)
elseif(NOT DEFINED CUDAToolkit_ROOT AND CHIMERA_TOOLCHAIN_ROOT AND EXISTS "${CHIMERA_TOOLCHAIN_ROOT}/bin/nvcc")
    set(CUDAToolkit_ROOT "${CHIMERA_TOOLCHAIN_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
endif()

if(CHIMERA_TOOLCHAIN_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${CHIMERA_TOOLCHAIN_ROOT}")
endif()

find_package(OpenMP REQUIRED)
find_package(cuvs REQUIRED)
find_package(rmm REQUIRED)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_compile_options(
    -fopenmp
    -march=native
    -ftree-vectorize
    -fexceptions
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_library(chimera_core STATIC
    cpp/third_party/rabitqlib/utils/fht_avx.cpp
    cpp/src/build_index.cu
    cpp/src/chimera_index.cu
    cpp/src/io.cpp
    cpp/src/ivf_pg.cpp
    cpp/src/kernels.cu
    cpp/src/quantization.cpp
    cpp/src/search_impl.cu
    cpp/src/utils.cu
)
target_include_directories(chimera_core PUBLIC
    cpp/include
    cpp/third_party
)
target_link_libraries(chimera_core PUBLIC
    OpenMP::OpenMP_CXX
    cuvs::cuvs
    rmm::rmm
)
set_target_properties(chimera_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(gpu_build
    cpp/apps/build.cpp
    cpp/apps/utils.cpp
)
target_include_directories(gpu_build PRIVATE
    cpp/apps
    cpp/src
)
set_target_properties(gpu_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_build PRIVATE chimera_core)

add_executable(gpu_search
    cpp/apps/search.cpp
    cpp/apps/utils.cpp
)
target_include_directories(gpu_search PRIVATE
    cpp/apps
    cpp/src
)
set_target_properties(gpu_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search PRIVATE chimera_core)

option(CHIMERA_BUILD_PYTHON "Build the Python extension module" ON)
if(CHIMERA_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 2.12 CONFIG REQUIRED)
    pybind11_add_module(chimera MODULE NO_EXTRAS python/bindings.cpp)
    target_link_libraries(chimera PRIVATE chimera_core)
    set_target_properties(chimera PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
