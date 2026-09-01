cmake_minimum_required(VERSION 3.24)
project(CHIMERA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

option(CHIMERA_ENABLE_GPU
    "Enable CUDA/cuVS-based GPU targets and backends"
    ON)

option(ENABLE_TEST
    "Build test_gather and test_dist_decomp targets"
    OFF)

option(CHIMERA_ENABLE_LEGACY_TARGETS
    "Build legacy Chimera search/build targets that are kept for source compatibility"
    OFF)

if(CHIMERA_ENABLE_GPU)
    add_compile_definitions(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

    if(NOT CHIMERA_TOOLCHAIN_ROOT)
        set(CHIMERA_TOOLCHAIN_ROOT "")
    endif()
    if(NOT CHIMERA_TOOLCHAIN_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
        set(CHIMERA_TOOLCHAIN_ROOT "$ENV{CONDA_PREFIX}")
    elseif(NOT CHIMERA_TOOLCHAIN_ROOT AND CMAKE_CUDA_COMPILER)
        get_filename_component(_chimera_nvcc_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        get_filename_component(_chimera_nvcc_root "${_chimera_nvcc_bin}" DIRECTORY)
        if(EXISTS "${_chimera_nvcc_root}/lib")
            set(CHIMERA_TOOLCHAIN_ROOT "${_chimera_nvcc_root}")
        endif()
    endif()

    if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
        set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "CUDA toolkit root" FORCE)
    elseif(NOT DEFINED CUDAToolkit_ROOT AND CHIMERA_TOOLCHAIN_ROOT AND EXISTS "${CHIMERA_TOOLCHAIN_ROOT}/bin/nvcc")
        set(CUDAToolkit_ROOT "${CHIMERA_TOOLCHAIN_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
    endif()

    if(CHIMERA_TOOLCHAIN_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${CHIMERA_TOOLCHAIN_ROOT}")
    endif()
endif()

find_package(OpenMP REQUIRED)

if(CHIMERA_ENABLE_GPU)
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

if(CHIMERA_ENABLE_GPU)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68,177,186>
    )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -Wno-deprecated-declarations")
endif()

option(CHIMERA_ENABLE_DETAILED_PROFILING
    "Compile detailed per-stage GPU profiling into regular search targets"
    OFF)

if(CHIMERA_ENABLE_DETAILED_PROFILING)
    add_compile_definitions(CHIMERA_PROFILE)
endif()
# add_compile_definitions(CHIMERA_TIMELINE)
add_compile_definitions(CHIMERA_OVERLAP_STAGE23)

option(CHIMERA_V6_GPU_INDICES_U32
    "Use uint32_t GPU token/cluster indices in v6; limits total token count to 4,294,967,295"
    OFF)

option(CHIMERA_CPU_V2_STAGE1_PREPACK
    "Prepack clustered Stage 1 data at startup for cpu_search_v2"
    ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    if(CHIMERA_ENABLE_GPU)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

# ---------- cpu library ----------
add_library(chimera_core STATIC
    cpp/src/estimator.cpp
    cpp/src/fht_avx.cpp
    cpp/src/index.cpp
    cpp/src/io.cpp
    cpp/src/ivf_pg.cpp
    cpp/src/quantization.cpp
    cpp/src/query.cpp
)
target_include_directories(chimera_core PUBLIC cpp/include)
target_include_directories(chimera_core PUBLIC cpp/dep)
target_link_libraries(chimera_core PUBLIC OpenMP::OpenMP_CXX)
if(CHIMERA_ENABLE_GPU)
    target_compile_definitions(chimera_core PUBLIC CHIMERA_HAVE_CUVS=1)
    target_link_libraries(chimera_core PUBLIC cuvs::cuvs rmm::rmm)
endif()

if(CHIMERA_ENABLE_LEGACY_TARGETS)
    include_directories(
        cpp/legacy/app
        cpp/legacy/include
        cpp/legacy/cuda/include
    )
endif()

# ---------- executables ----------
add_library(chimera_app_utils STATIC cpp/app/utils.cpp)
target_include_directories(chimera_app_utils PUBLIC cpp/app)

if(CHIMERA_ENABLE_LEGACY_TARGETS)
    add_executable(cpu_build cpp/app/cpu_build.cpp)
    target_link_libraries(cpu_build PRIVATE chimera_core)

    add_executable(cpu_search_v1
        cpp/legacy/app/cpu_search_v1.cpp
        cpp/legacy/src/cpu_kernel_v1.cpp
    )
    target_link_libraries(cpu_search_v1 PRIVATE chimera_core chimera_app_utils)

    add_executable(cpu_search_v2
        cpp/legacy/app/cpu_search_v2.cpp
        cpp/legacy/src/cpu_kernel_v2.cpp
        cpp/legacy/src/cpu_kernel_v1.cpp
    )
    target_compile_definitions(cpu_search_v2 PRIVATE
        CHIMERA_CPU_V2_STAGE1_PREPACK=$<BOOL:${CHIMERA_CPU_V2_STAGE1_PREPACK}>
    )
    target_link_libraries(cpu_search_v2 PRIVATE chimera_core chimera_app_utils)
endif()

add_executable(cpu_search
    cpp/app/cpu_search.cpp
    cpp/src/cpu_kernel.cpp
)
target_compile_definitions(cpu_search PRIVATE
    CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE=1
    CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE=1
)
target_link_libraries(cpu_search PRIVATE chimera_core chimera_app_utils)

if(CHIMERA_ENABLE_LEGACY_TARGETS)
    add_executable(cpu_search_parallel
        cpp/app/cpu_search.cpp
        cpp/src/cpu_kernel.cpp
    )
    target_compile_definitions(cpu_search_parallel PRIVATE
        CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE=2
        CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE=0
    )
    target_link_libraries(cpu_search_parallel PRIVATE chimera_core chimera_app_utils)

    add_executable(cpu_search_parallel_dynamic
        cpp/app/cpu_search.cpp
        cpp/src/cpu_kernel.cpp
    )
    target_compile_definitions(cpu_search_parallel_dynamic PRIVATE
        CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE=2
        CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE=1
    )
    target_link_libraries(cpu_search_parallel_dynamic PRIVATE chimera_core chimera_app_utils)

    add_executable(cpu_search_hybrid
        cpp/app/cpu_search.cpp
        cpp/src/cpu_kernel.cpp
    )
    target_compile_definitions(cpu_search_hybrid PRIVATE
        CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE=3
        CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE=0
    )
    target_link_libraries(cpu_search_hybrid PRIVATE chimera_core chimera_app_utils)

    add_executable(cpu_search_hybrid_dynamic
        cpp/app/cpu_search.cpp
        cpp/src/cpu_kernel.cpp
    )
    target_compile_definitions(cpu_search_hybrid_dynamic PRIVATE
        CHIMERA_CPU_STAGE1_DOC_ACCUM_MODE=3
        CHIMERA_CPU_STAGE3_DYNAMIC_SCHEDULE=1
    )
    target_link_libraries(cpu_search_hybrid_dynamic PRIVATE chimera_core chimera_app_utils)
endif()

if(ENABLE_TEST)
    add_executable(test_gather cpp/test/test_gather.cpp)
    target_include_directories(test_gather PRIVATE cpp/app)
    target_link_libraries(test_gather PRIVATE chimera_core chimera_app_utils)

    add_executable(test_dist_decomp cpp/test/test_dist_decomp.cpp)
    target_include_directories(test_dist_decomp PRIVATE cpp/app)
    target_link_libraries(test_dist_decomp PRIVATE chimera_core chimera_app_utils)
endif()

if(CHIMERA_ENABLE_GPU)
if(CHIMERA_ENABLE_LEGACY_TARGETS)
add_executable(gpu_search_v3
    cpp/legacy/app/gpu_search_v3.cu
    cpp/legacy/cuda/src/gpu_index_v3.cu
    cpp/cuda/src/gpu_kernels_base.cu
)
target_include_directories(gpu_search_v3 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v3 PRIVATE
    CHIMERA_IMPL_NAME=\"v3\"
    CHIMERA_USE_LUT=1
)
target_link_libraries(gpu_search_v3 PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v7
    cpp/legacy/app/gpu_search_v7.cu
    cpp/legacy/cuda/src/gpu_index_v7.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/legacy/cuda/src/gpu_kernels_v7.cu
)
target_include_directories(gpu_search_v7 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v7 PRIVATE
    CHIMERA_IMPL_NAME=\"v7\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
)
target_link_libraries(gpu_search_v7 PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v7_lite
    cpp/legacy/app/gpu_search_v7_lite.cu
    cpp/legacy/cuda/src/gpu_index_v7_lite.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/legacy/cuda/src/gpu_kernels_v7_lite.cu
)
target_include_directories(gpu_search_v7_lite PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v7_lite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v7_lite PRIVATE
    CHIMERA_IMPL_NAME=\"v7_lite\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
)
target_link_libraries(gpu_search_v7_lite PRIVATE chimera_core chimera_app_utils)
add_custom_target(v7_lite DEPENDS gpu_search_v7_lite)

add_executable(gpu_search_v7_unalign
    cpp/legacy/app/gpu_search_v7_lite.cu
    cpp/legacy/cuda/src/gpu_index_v7_lite.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/legacy/cuda/src/gpu_kernels_v7_lite.cu
)
target_include_directories(gpu_search_v7_unalign PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v7_unalign PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v7_unalign PRIVATE
    CHIMERA_IMPL_NAME=\"v7_unalign\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
    CHIMERA_V7_DISABLE_STAGE1_SHIFT=1
)
target_link_libraries(gpu_search_v7_unalign PRIVATE chimera_core chimera_app_utils)
add_custom_target(v7_unalign DEPENDS gpu_search_v7_unalign)
endif()

add_executable(gpu_search
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search PRIVATE
    CHIMERA_IMPL_NAME=\"v8\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
)
target_link_libraries(gpu_search PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_nosum
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search_nosum PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_nosum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_nosum PRIVATE
    CHIMERA_IMPL_NAME=\"v8_nosum\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
    CHIMERA_STAGE2_USE_V0_DOC_SCORE=1
)
target_link_libraries(gpu_search_nosum PRIVATE chimera_core chimera_app_utils)
add_custom_target(v8_nosum DEPENDS gpu_search_nosum)

if(CHIMERA_ENABLE_LEGACY_TARGETS)
add_executable(gpu_search_v9
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search_v9 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v9 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v9 PRIVATE
    CHIMERA_IMPL_NAME=\"v9\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
    CHIMERA_STAGE2_DOC_LAYOUT=1
)
target_link_libraries(gpu_search_v9 PRIVATE chimera_core chimera_app_utils)
add_custom_target(v9 DEPENDS gpu_search_v9)
endif()

add_executable(gpu_search_nolut
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search_nolut PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_nolut PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_nolut PRIVATE
    CHIMERA_IMPL_NAME=\"v8_nolut\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
)
target_link_libraries(gpu_search_nolut PRIVATE chimera_core chimera_app_utils)
add_custom_target(v8_nolut DEPENDS gpu_search_nolut)
add_custom_target(v8s8_nolut DEPENDS gpu_search_nolut)

add_executable(gpu_search_nolut_nosum
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search_nolut_nosum PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_nolut_nosum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_nolut_nosum PRIVATE
    CHIMERA_IMPL_NAME=\"v8_nolut_nosum\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_STAGE2_USE_V0_DOC_SCORE=1
)
target_link_libraries(gpu_search_nolut_nosum PRIVATE chimera_core chimera_app_utils)
add_custom_target(v8_nolut_nosum DEPENDS gpu_search_nolut_nosum)
add_custom_target(v8s8_nolut_nosum DEPENDS gpu_search_nolut_nosum)

add_executable(gpu_search_profile
    cpp/app/gpu_search.cu
    cpp/cuda/src/gpu_index.cu
    cpp/cuda/src/gpu_kernels_base.cu
    cpp/cuda/src/gpu_kernels.cu
)
target_include_directories(gpu_search_profile PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_profile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_profile PRIVATE
    CHIMERA_IMPL_NAME=\"v8_profile\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
    CHIMERA_PROFILE=1
    CHIMERA_PROFILE_APP=1
)
target_link_libraries(gpu_search_profile PRIVATE chimera_core chimera_app_utils)

if(CHIMERA_ENABLE_LEGACY_TARGETS)
function(add_gpu_search_v4_nonclustered_target target_name impl_name)
    add_executable(${target_name}
        cpp/legacy/app/gpu_search_v4.cu
        cpp/legacy/cuda/src/gpu_index_v4.cu
        cpp/legacy/cuda/src/gpu_kernels_v4.cu
    )
    target_include_directories(${target_name} PRIVATE
        cpp/include
        cpp/cuda/include
    )
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target_name} PRIVATE
        CHIMERA_IMPL_NAME=\"${impl_name}\"
        CHIMERA_USE_LUT=1
        ${ARGN}
    )
    target_link_libraries(${target_name} PRIVATE chimera_core chimera_app_utils)
endfunction()

add_gpu_search_v4_nonclustered_target(
    gpu_search_v4
    v4
)

add_executable(gpu_search_v5
    cpp/legacy/app/gpu_search_v5.cu
    cpp/legacy/cuda/src/gpu_index_v5.cu
    cpp/legacy/cuda/src/gpu_kernels_v5.cu
)
target_include_directories(gpu_search_v5 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v5 PRIVATE
    CHIMERA_IMPL_NAME=\"v5\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
)
target_link_libraries(gpu_search_v5 PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v5_lite
    cpp/legacy/app/gpu_search_v6_lut.cu
    cpp/legacy/cuda/src/gpu_index_v6_lut.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_lut.cu
)
target_include_directories(gpu_search_v5_lite PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v5_lite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v5_lite PRIVATE
    CHIMERA_IMPL_NAME=\"v5_lite\"
    CHIMERA_CLUSTERED_LAYOUT_REQUIRED=1
    CHIMERA_USE_LUT=1
    CHIMERA_COMPACT_DOC_BUFFER=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v5_lite PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v5_lite PRIVATE chimera_core chimera_app_utils)
add_custom_target(v5_lite DEPENDS gpu_search_v5_lite)

add_executable(gpu_search_v6
    cpp/legacy/app/gpu_search_v6_lut.cu
    cpp/legacy/cuda/src/gpu_index_v6_lut.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_lut.cu
)
target_include_directories(gpu_search_v6 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6 PRIVATE
    CHIMERA_IMPL_NAME=\"v6\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6 PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6 PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v6_nosum
    cpp/legacy/app/gpu_search_v6_nosum.cu
    cpp/legacy/cuda/src/gpu_index_v6_nosum.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_lut.cu
    cpp/legacy/cuda/src/gpu_kernels_v0.cu
)
target_include_directories(gpu_search_v6_nosum PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_nosum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_nosum PRIVATE
    CHIMERA_IMPL_NAME=\"v6_nosum\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_nosum PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6_nosum PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v6_lite
    cpp/legacy/app/gpu_search_v6_lite.cu
    cpp/legacy/cuda/src/gpu_index_v6_lite.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_lite.cu
)
target_include_directories(gpu_search_v6_lite PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_lite PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_lite PRIVATE
    CHIMERA_IMPL_NAME=\"v6_lite\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
    CHIMERA_V6_FORCE_DOCPTRS=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_lite PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6_lite PRIVATE chimera_core chimera_app_utils)
add_custom_target(v6_lite DEPENDS gpu_search_v6_lite)

add_executable(gpu_search_v6_turbo
    cpp/legacy/app/gpu_search_v6_turbo.cu
    cpp/legacy/cuda/src/gpu_index_v6_turbo.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_lite.cu
)
target_include_directories(gpu_search_v6_turbo PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_turbo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_turbo PRIVATE
    CHIMERA_IMPL_NAME=\"v6_turbo\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
    CHIMERA_V6_FORCE_DOCPTRS=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_turbo PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6_turbo PRIVATE chimera_core chimera_app_utils)
add_custom_target(v6_turbo DEPENDS gpu_search_v6_turbo)

add_executable(gpu_search_v6_nolut
    cpp/legacy/app/gpu_search_v6_nolut.cu
    cpp/legacy/cuda/src/gpu_index_v6_nolut.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_nolut.cu
)
target_include_directories(gpu_search_v6_nolut PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_nolut PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_nolut PRIVATE
    CHIMERA_IMPL_NAME=\"v6_nolut\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_nolut PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6_nolut PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v6_nolut_nosum
    cpp/legacy/app/gpu_search_v6_nolut_nosum.cu
    cpp/legacy/cuda/src/gpu_index_v6_nolut_nosum.cu
    cpp/legacy/cuda/src/gpu_kernels_v6_nolut.cu
    cpp/legacy/cuda/src/gpu_kernels_v0.cu
)
target_include_directories(gpu_search_v6_nolut_nosum PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v6_nolut_nosum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(gpu_search_v6_nolut_nosum PRIVATE
    CHIMERA_IMPL_NAME=\"v6_nolut_nosum\"
    CHIMERA_CLUSTERED_LAYOUT_DISABLED=1
    CHIMERA_COMPACT_DOC_BUFFER=1
)
if(CHIMERA_V6_GPU_INDICES_U32)
    target_compile_definitions(gpu_search_v6_nolut_nosum PRIVATE
        CHIMERA_V6_GPU_INDICES_U32=1
    )
endif()
target_link_libraries(gpu_search_v6_nolut_nosum PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v2
    cpp/legacy/app/gpu_search_v2.cu
    cpp/legacy/cuda/src/gpu_index_v2.cu
    cpp/legacy/cuda/src/gpu_kernels_v2.cu
)
target_include_directories(gpu_search_v2 PRIVATE
    cpp/include
    cpp/cuda/include
)
target_compile_definitions(gpu_search_v2 PRIVATE
    CHIMERA_IMPL_NAME=\"v2\"
    CHIMERA_DOC_SCORE_IMPL=2
    CHIMERA_BINARY_IP_IMPL=2
    CHIMERA_CPU_REFINE_IMPL=2
    CHIMERA_STAGE1_SPARSE_CLEANUP=0
)
set_target_properties(gpu_search_v2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v2 PRIVATE chimera_core chimera_app_utils)

add_executable(gpu_search_v1
    cpp/legacy/app/gpu_search_v1.cu
    cpp/legacy/cuda/src/gpu_index_v1.cu
    cpp/legacy/cuda/src/gpu_kernels_v1.cu
)
target_include_directories(gpu_search_v1 PRIVATE
    cpp/include
    cpp/cuda/include
)
target_compile_definitions(gpu_search_v1 PRIVATE
    CHIMERA_IMPL_NAME=\"v1\"
    CHIMERA_DOC_SCORE_IMPL=1
    CHIMERA_BINARY_IP_IMPL=1
    CHIMERA_CPU_REFINE_IMPL=1
    CHIMERA_STAGE1_SPARSE_CLEANUP=0
)
set_target_properties(gpu_search_v1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v1 PRIVATE chimera_core chimera_app_utils)
endif()

add_executable(gpu_build
    cpp/app/gpu_build.cpp
    cpp/cuda/src/build_index.cu
)
target_include_directories(gpu_build PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_build PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_build PRIVATE chimera_core)

if(CHIMERA_ENABLE_LEGACY_TARGETS)
add_executable(gpu_search_v0
    cpp/legacy/app/gpu_search_v0.cu
    cpp/legacy/cuda/src/gpu_index_v0.cu
    cpp/legacy/cuda/src/gpu_kernels_v0.cu
)
target_include_directories(gpu_search_v0 PRIVATE
    cpp/include
    cpp/cuda/include
)
set_target_properties(gpu_search_v0 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpu_search_v0 PRIVATE chimera_core chimera_app_utils)
endif()
endif()
