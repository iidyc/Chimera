list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES GPU_MVR_CONDA_PREFIX)

if(DEFINED GPU_MVR_CONDA_PREFIX)
    set(GPU_MVR_CONDA_PREFIX "${GPU_MVR_CONDA_PREFIX}" CACHE PATH "Explicit Conda environment")
endif()

if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
    set(_gpu_mvr_conda_prefix "$ENV{CONDA_PREFIX}")
elseif(DEFINED GPU_MVR_CONDA_PREFIX AND EXISTS "${GPU_MVR_CONDA_PREFIX}")
    set(_gpu_mvr_conda_prefix "${GPU_MVR_CONDA_PREFIX}")
else()
    message(FATAL_ERROR
        "A Conda environment is required. Activate the target environment first, "
        "or configure with -DGPU_MVR_CONDA_PREFIX=/path/to/env."
    )
endif()

if(NOT EXISTS "${_gpu_mvr_conda_prefix}/bin/nvcc")
    message(FATAL_ERROR
        "Could not find nvcc in ${_gpu_mvr_conda_prefix}/bin. "
        "Install the CUDA toolkit into the Conda environment first."
    )
endif()

set(CMAKE_CUDA_COMPILER "${_gpu_mvr_conda_prefix}/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CUDAToolkit_ROOT "${_gpu_mvr_conda_prefix}" CACHE PATH "CUDA toolkit root")
set(CMAKE_PREFIX_PATH "${_gpu_mvr_conda_prefix};${CMAKE_PREFIX_PATH}" CACHE STRING "Package search roots" FORCE)
set(GPU_MVR_CONDA_TOOLCHAIN ON CACHE BOOL "Build with Conda-first toolchain")
set(GPU_MVR_TOOLCHAIN_ROOT "${_gpu_mvr_conda_prefix}" CACHE PATH "Resolved Conda toolchain root")

if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "CUDA runtime library")
endif()

if(NOT DEFINED CMAKE_BUILD_RPATH)
    set(CMAKE_BUILD_RPATH "${_gpu_mvr_conda_prefix}/lib" CACHE STRING "Build RPATH")
endif()

if(NOT DEFINED CMAKE_INSTALL_RPATH)
    set(CMAKE_INSTALL_RPATH "${_gpu_mvr_conda_prefix}/lib" CACHE STRING "Install RPATH")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Use link path for install RPATH")
set(_gpu_mvr_link_flags
    "-L${_gpu_mvr_conda_prefix}/lib -Wl,-rpath,${_gpu_mvr_conda_prefix}/lib -Wl,-rpath-link,${_gpu_mvr_conda_prefix}/lib"
)
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_gpu_mvr_link_flags}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_gpu_mvr_link_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_gpu_mvr_link_flags}")