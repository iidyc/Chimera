cmake_minimum_required(VERSION 3.24)
project(GPU_MVR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

option(GPU_MVR_ENABLE_GPU
    "Enable CUDA/cuVS-based GPU targets and backends"
    ON)

option(ENABLE_TEST
    "Build test_gather and test_dist_decomp targets"
    OFF)

if(GPU_MVR_ENABLE_GPU)
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
endif()

find_package(OpenMP REQUIRED)

if(GPU_MVR_ENABLE_GPU)
    find_package(cuvs REQUIRED)
    find_package(rmm REQUIRED)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

add_compile_options(
    -fopenmp
    -march=native
    -ftree-vectorize
    -fexceptions
)

if(GPU_MVR_ENABLE_GPU)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations")
endif()

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
    if(GPU_MVR_ENABLE_GPU)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
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
target_include_directories(gpu_mvr_core PUBLIC cpp/dep)
target_link_libraries(gpu_mvr_core PUBLIC OpenMP::OpenMP_CXX)
if(GPU_MVR_ENABLE_GPU)
    target_compile_definitions(gpu_mvr_core PUBLIC GPU_MVR_HAVE_CUVS=1)
    target_link_libraries(gpu_mvr_core PUBLIC cuvs::cuvs rmm::rmm)
endif()

# ---------- executables ----------
add_library(gpu_mvr_app_utils STATIC cpp/app/utils.cpp)
target_include_directories(gpu_mvr_app_utils PUBLIC cpp/app)

add_executable(cpu_build cpp/app/cpu_build.cpp)
target_link_libraries(cpu_build PRIVATE gpu_mvr_core)

add_executable(cpu_search_v1
    cpp/app/cpu_search_v1.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_link_libraries(cpu_search_v1 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v2
    cpp/app/cpu_search_v2.cpp
    cpp/src/cpu_kernel_v2.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v2 PRIVATE
    GPU_MVR_CPU_V2_STAGE1_PREPACK=$<BOOL:${GPU_MVR_CPU_V2_STAGE1_PREPACK}>
)
target_link_libraries(cpu_search_v2 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v3
    cpp/app/cpu_search_v3.cpp
    cpp/src/cpu_kernel_v3.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v3 PRIVATE
    GPU_MVR_CPU_V3_STAGE1_DOC_ACCUM_MODE=1
    GPU_MVR_CPU_V3_STAGE3_DYNAMIC_SCHEDULE=1
)
target_link_libraries(cpu_search_v3 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v3_parallel
    cpp/app/cpu_search_v3.cpp
    cpp/src/cpu_kernel_v3.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v3_parallel PRIVATE
    GPU_MVR_CPU_V3_STAGE1_DOC_ACCUM_MODE=2
    GPU_MVR_CPU_V3_STAGE3_DYNAMIC_SCHEDULE=0
)
target_link_libraries(cpu_search_v3_parallel PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v3_parallel_dynamic
    cpp/app/cpu_search_v3.cpp
    cpp/src/cpu_kernel_v3.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v3_parallel_dynamic PRIVATE
    GPU_MVR_CPU_V3_STAGE1_DOC_ACCUM_MODE=2
    GPU_MVR_CPU_V3_STAGE3_DYNAMIC_SCHEDULE=1
)
target_link_libraries(cpu_search_v3_parallel_dynamic PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v3_hybrid
    cpp/app/cpu_search_v3.cpp
    cpp/src/cpu_kernel_v3.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v3_hybrid PRIVATE
    GPU_MVR_CPU_V3_STAGE1_DOC_ACCUM_MODE=3
    GPU_MVR_CPU_V3_STAGE3_DYNAMIC_SCHEDULE=0
)
target_link_libraries(cpu_search_v3_hybrid PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(cpu_search_v3_hybrid_dynamic
    cpp/app/cpu_search_v3.cpp
    cpp/src/cpu_kernel_v3.cpp
    cpp/src/cpu_kernel_v1.cpp
)
target_compile_definitions(cpu_search_v3_hybrid_dynamic PRIVATE
    GPU_MVR_CPU_V3_STAGE1_DOC_ACCUM_MODE=3
    GPU_MVR_CPU_V3_STAGE3_DYNAMIC_SCHEDULE=1
)
target_link_libraries(cpu_search_v3_hybrid_dynamic PRIVATE gpu_mvr_core gpu_mvr_app_utils)

if(ENABLE_TEST)
    add_executable(test_gather cpp/test/test_gather.cpp)
    target_include_directories(test_gather PRIVATE cpp/app)
    target_link_libraries(test_gather PRIVATE gpu_mvr_core gpu_mvr_app_utils)

    add_executable(test_dist_decomp cpp/test/test_dist_decomp.cpp)
    target_include_directories(test_dist_decomp PRIVATE cpp/app)
    target_link_libraries(test_dist_decomp PRIVATE gpu_mvr_core gpu_mvr_app_utils)
endif()

if(GPU_MVR_ENABLE_GPU)
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
target_link_libraries(gpu_search_v3 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(gpu_search_v7
    cpp/app/gpu_search_v7.cu
    cpp/cuda/src/gpu_index_v7.cu
    cpp/cuda/src/gpu_kernels_v3.cu
    cpp/cuda/src/gpu_kernels_v7.cu
)
target_include_directories(gpu_search_v7 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v7 PRIVATE
    GPU_MVR_IMPL_NAME=\"v7\"
    GPU_MVR_CLUSTERED_LAYOUT_REQUIRED=1
)
target_link_libraries(gpu_search_v7 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(gpu_search_v8
    cpp/app/gpu_search_v8.cu
    cpp/cuda/src/gpu_index_v8.cu
    cpp/cuda/src/gpu_kernels_v3.cu
    cpp/cuda/src/gpu_kernels_v8.cu
)
target_include_directories(gpu_search_v8 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v8 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v8 PRIVATE
    GPU_MVR_IMPL_NAME=\"v8\"
    GPU_MVR_CLUSTERED_LAYOUT_REQUIRED=1
)
target_link_libraries(gpu_search_v8 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(gpu_search_v8_nolut
    cpp/app/gpu_search_v8.cu
    cpp/cuda/src/gpu_index_v8.cu
    cpp/cuda/src/gpu_kernels_v3.cu
    cpp/cuda/src/gpu_kernels_v8.cu
)
target_include_directories(gpu_search_v8_nolut PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v8_nolut PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v8_nolut PRIVATE
    GPU_MVR_IMPL_NAME=\"v8_nolut\"
    GPU_MVR_CLUSTERED_LAYOUT_REQUIRED=1
)
target_compile_options(gpu_search_v8_nolut PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-UGPU_MVR_USE_LUT>
    $<$<COMPILE_LANGUAGE:CUDA>:-UGPU_MVR_USE_LUT>
)
target_link_libraries(gpu_search_v8_nolut PRIVATE gpu_mvr_core gpu_mvr_app_utils)

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
    target_link_libraries(${target_name} PRIVATE gpu_mvr_core gpu_mvr_app_utils)
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
target_link_libraries(gpu_search_v5 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

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
target_link_libraries(gpu_search_v6 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(gpu_search_v6_nolut
    cpp/app/gpu_search_v6.cu
    cpp/cuda/src/gpu_index_v6.cu
    cpp/cuda/src/gpu_kernels_v6.cu
)
target_include_directories(gpu_search_v6_nolut PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_nolut PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_nolut PRIVATE
    GPU_MVR_IMPL_NAME=\"v6_nolut\"
    GPU_MVR_CLUSTERED_LAYOUT_DISABLED=1
)
if(GPU_MVR_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_nolut PRIVATE
        GPU_MVR_V6_GPU_INDICES_U32=1
    )
endif()
target_compile_options(gpu_search_v6_nolut PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-UGPU_MVR_USE_LUT -UGPU_MVR_COMPACT_DOC_BUFFER>
    $<$<COMPILE_LANGUAGE:CUDA>:-UGPU_MVR_USE_LUT -UGPU_MVR_COMPACT_DOC_BUFFER>
)
target_link_libraries(gpu_search_v6_nolut PRIVATE gpu_mvr_core gpu_mvr_app_utils)

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
target_link_libraries(gpu_search_v2 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

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
target_link_libraries(gpu_search_v1 PRIVATE gpu_mvr_core gpu_mvr_app_utils)

add_executable(gpu_build 
    cpp/app/gpu_build.cpp
    cpp/cuda/src/build_index.cu
)
target_include_directories(gpu_build PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_build PRIVATE gpu_mvr_core)

add_executable(gpu_search_v0
    cpp/app/gpu_search_v0.cu
    cpp/cuda/src/gpu_index_v0.cu
    cpp/cuda/src/gpu_kernels_v0.cu
)
target_include_directories(gpu_search_v0 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v0 PRIVATE gpu_mvr_core gpu_mvr_app_utils)
endif()
