#define GPU_MVR_CPU_V2_STAGE1_PREPACK 1
#define cpu_kernel_v2 cpu_kernel_v3
#include "cpu_kernel_v2.cpp"
#undef cpu_kernel_v2
#undef GPU_MVR_CPU_V2_STAGE1_PREPACK
