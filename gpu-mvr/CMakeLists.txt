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
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_compile_options(
    -fopenmp
    -march=native
    -ftree-vectorize
    -fexceptions
)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations")

add_compile_definitions(GPU_MVR_PROFILE)
# add_compile_definitions(GPU_MVR_TIMELINE)
add_compile_definitions(GPU_MVR_OVERLAP_STAGE23)
add_compile_definitions(GPU_MVR_USE_LUT)

option(GPU_MVR_V6_GPU_INDICES_U32
    "Use uint32_t GPU token/cluster indices in v6; limits total token count to 4,294,967,295"
    OFF)

option(GPU_MVR_CPU_V2_STAGE1_PREPACK
    "Prepack clustered Stage 1 data at startup for cpu_search_v2"
    ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# ---------- cpu library ----------
add_library(gpu_mvr_core STATIC
    cpp/src/estimator.cpp
    cpp/src/fht_avx.cpp
    cpp/src/index.cpp
    cpp/src/io.cpp
    cpp/src/ivf_pg.cpp
    cpp/src/quantization.cpp
    cpp/src/query.cpp
)
target_include_directories(gpu_mvr_core PUBLIC cpp/include)
target_link_libraries(gpu_mvr_core PUBLIC cuvs::cuvs OpenMP::OpenMP_CXX rmm::rmm)

# ---------- executables ----------
# utils.hpp lives in cpp/app/ — add as include dir for all app targets
# include_directories(cpp/app)

add_executable(build cpp/app/build.cpp)
target_link_libraries(build PRIVATE gpu_mvr_core)

add_executable(search cpp/app/search.cpp)
target_link_libraries(search PRIVATE gpu_mvr_core)

add_executable(cpu_search_v1
    cpp/app/cpu_search_v1.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_link_libraries(cpu_search_v1 PRIVATE gpu_mvr_core)

add_executable(cpu_search_v2
    cpp/app/cpu_search_v2.cpp
    cpp/src/cpu_kernel_v2.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v2 PRIVATE
    GPU_MVR_CPU_V2_STAGE1_PREPACK=$<BOOL:${GPU_MVR_CPU_V2_STAGE1_PREPACK}>
)
target_link_libraries(cpu_search_v2 PRIVATE gpu_mvr_core)

add_executable(test_gather cpp/app/test_gather.cpp)
target_link_libraries(test_gather PRIVATE gpu_mvr_core)

add_executable(test_dist_decomp cpp/app/test_dist_decomp.cpp)
target_link_libraries(test_dist_decomp PRIVATE gpu_mvr_core)

add_executable(build_cagra cpp/app/build_cagra.cpp)
target_link_libraries(build_cagra PRIVATE gpu_mvr_core)

add_executable(build_hnsw_from_cagra cpp/app/build_hnsw_from_cagra.cpp)
target_link_libraries(build_hnsw_from_cagra PRIVATE gpu_mvr_core)

add_executable(gpu_search
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search PRIVATE GPU_MVR_IMPL_NAME=\"v3\")
target_link_libraries(gpu_search PRIVATE gpu_mvr_core)

add_executable(gpu_search_v3
    cpp/app/gpu_search_v3.cu
    cpp/cuda/src/gpu_index_v3.cu
    cpp/cuda/src/gpu_kernels_v3.cu
)
target_include_directories(gpu_search_v3 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v3 PRIVATE GPU_MVR_IMPL_NAME=\"v3\")
target_link_libraries(gpu_search_v3 PRIVATE gpu_mvr_core)

function(add_gpu_search_v4_nonclustered_target target_name impl_name)
    add_executable(${target_name}
        cpp/app/gpu_search_v4.cu
        cpp/cuda/src/gpu_index_v4.cu
        cpp/cuda/src/gpu_kernels_v4.cu
    )
    target_include_directories(${target_name} PRIVATE
        cpp/include
        cpp/cuda/include
    )
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target_name} PRIVATE
        GPU_MVR_IMPL_NAME=\"${impl_name}\"
        ${ARGN}
    )
    target_link_libraries(${target_name} PRIVATE gpu_mvr_core)
endfunction()

add_gpu_search_v4_nonclustered_target(
    gpu_search_v4
    v4
)

add_executable(gpu_search_v5
    cpp/app/gpu_search_v5.cu
    cpp/cuda/src/gpu_index_v5.cu
    cpp/cuda/src/gpu_kernels_v5.cu
)
target_include_directories(gpu_search_v5 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v5 PRIVATE
    GPU_MVR_IMPL_NAME=\"v5\"
    GPU_MVR_CLUSTERED_LAYOUT_REQUIRED=1
)
target_link_libraries(gpu_search_v5 PRIVATE gpu_mvr_core)

add_executable(gpu_search_v6
    cpp/app/gpu_search_v6.cu
    cpp/cuda/src/gpu_index_v6.cu
    cpp/cuda/src/gpu_kernels_v6.cu
)
target_include_directories(gpu_search_v6 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6 PRIVATE
    GPU_MVR_IMPL_NAME=\"v6\"
    GPU_MVR_CLUSTERED_LAYOUT_DISABLED=1
    GPU_MVR_COMPACT_DOC_BUFFER=1
)
if(GPU_MVR_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6 PRIVATE
        GPU_MVR_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6 PRIVATE gpu_mvr_core)

add_executable(gpu_search_v2
    cpp/app/gpu_search_v2.cu
    cpp/cuda/src/gpu_index_v2.cu
    cpp/cuda/src/gpu_kernels_v2.cu
)
target_include_directories(gpu_search_v2 PRIVATE
    cpp/include
    cpp/cuda/include
)
target_compile_definitions(gpu_search_v2 PRIVATE
    GPU_MVR_IMPL_NAME=\"v2\"
    GPU_MVR_DOC_SCORE_IMPL=2
    GPU_MVR_BINARY_IP_IMPL=2
    GPU_MVR_CPU_REFINE_IMPL=2
    GPU_MVR_STAGE1_SPARSE_CLEANUP=0
)
set_target_properties(gpu_search_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v2 PRIVATE gpu_mvr_core)

add_executable(gpu_search_v1
    cpp/app/gpu_search_v1.cu
    cpp/cuda/src/gpu_index_v1.cu
    cpp/cuda/src/gpu_kernels_v1.cu
)
target_include_directories(gpu_search_v1 PRIVATE
    cpp/include
    cpp/cuda/include
)
target_compile_definitions(gpu_search_v1 PRIVATE
    GPU_MVR_IMPL_NAME=\"v1\"
    GPU_MVR_DOC_SCORE_IMPL=1
    GPU_MVR_BINARY_IP_IMPL=1
    GPU_MVR_CPU_REFINE_IMPL=1
    GPU_MVR_STAGE1_SPARSE_CLEANUP=0
)
set_target_properties(gpu_search_v1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v1 PRIVATE gpu_mvr_core)

add_executable(gpu_build 
    cpp/app/gpu_build.cu
    cpp/cuda/src/build_gpu_index.cu
)
target_include_directories(gpu_build PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_build PRIVATE gpu_mvr_core)

add_executable(gpu_build_fast
    cpp/app/gpu_build_fast.cpp
    cpp/cuda/src/build_gpu_index_fast.cu
)
target_include_directories(gpu_build_fast PRIVATE
    cpp/include
    cpp/cuda/include
)
set_source_files_properties(
    cpp/cuda/src/build_gpu_index_fast.cu
    PROPERTIES LANGUAGE CXX
)
target_link_libraries(gpu_build_fast PRIVATE gpu_mvr_core)

add_executable(gpu_build_doc_4bit_ex
    cpp/app/gpu_build_doc_4bit_ex.cpp
)
target_include_directories(gpu_build_doc_4bit_ex PRIVATE
    cpp/include
    cpp/cuda/include
)
target_link_libraries(gpu_build_doc_4bit_ex PRIVATE gpu_mvr_core)

add_executable(gpu_search_v0
    cpp/app/gpu_search_baseline.cu
    cpp/cuda/src/gpu_index_baseline.cu
    cpp/cuda/src/gpu_kernels_baseline.cu
)
target_include_directories(gpu_search_v0 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v0 PRIVATE gpu_mvr_core)
