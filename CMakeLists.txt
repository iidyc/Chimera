cmake_minimum_required(VERSION 3.24)
project(GPU_MVR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

if(NOT GPU_MVR_TOOLCHAIN_ROOT)
    set(GPU_MVR_TOOLCHAIN_ROOT "")
endif()
if(NOT GPU_MVR_TOOLCHAIN_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
    set(GPU_MVR_TOOLCHAIN_ROOT "$ENV{CONDA_PREFIX}")
elseif(NOT GPU_MVR_TOOLCHAIN_ROOT AND CMAKE_CUDA_COMPILER)
    get_filename_component(_gpu_mvr_nvcc_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(_gpu_mvr_nvcc_root "${_gpu_mvr_nvcc_bin}" DIRECTORY)
    if(EXISTS "${_gpu_mvr_nvcc_root}/lib")
        set(GPU_MVR_TOOLCHAIN_ROOT "${_gpu_mvr_nvcc_root}")
    endif()
endif()

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "CUDA toolkit root" FORCE)
elseif(NOT DEFINED CUDAToolkit_ROOT AND GPU_MVR_TOOLCHAIN_ROOT AND EXISTS "${GPU_MVR_TOOLCHAIN_ROOT}/bin/nvcc")
    set(CUDAToolkit_ROOT "${GPU_MVR_TOOLCHAIN_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
endif()

if(GPU_MVR_TOOLCHAIN_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${GPU_MVR_TOOLCHAIN_ROOT}")
endif()

find_package(cuvs REQUIRED)
find_package(rmm REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

add_compile_options(
    -fopenmp
    -march=native
    -ftree-vectorize
    -fexceptions
)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
)

add_compile_definitions(GPU_MVR_PROFILE)
# add_compile_definitions(GPU_MVR_TIMELINE)
add_compile_definitions(GPU_MVR_OVERLAP_STAGE23)
add_compile_definitions(GPU_MVR_USE_LUT)

include_directories("include")

add_executable(build expr/build.cpp)
target_link_libraries(build PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(search expr/search.cpp)
target_link_libraries(search PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(test_gather expr/test_gather.cpp)
target_link_libraries(test_gather PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(test_dist_decomp expr/test_dist_decomp.cpp)
target_link_libraries(test_dist_decomp PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(build_cagra expr/build_cagra.cpp)
target_link_libraries(build_cagra PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(gpu_search expr/gpu_search.cu)
set_target_properties(gpu_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(gpu_build expr/gpu_build.cu)
set_target_properties(gpu_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_build PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(gpu_search_baseline expr/gpu_search_baseline.cu)
set_target_properties(gpu_search_baseline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_baseline PRIVATE cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

add_executable(build_ivfdockmeans expr/build_ivfdockmeans.cpp)
target_link_libraries(build_ivfdockmeans PRIVATE OpenMP::OpenMP_CXX)
