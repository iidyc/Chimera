#pragma once

#include <cstdint>

#ifdef GPU_MVR_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>

namespace gpu_mvr::profile {

class ScopedNvtxRange {
public:
    ScopedNvtxRange(const char* name, std::uint32_t argb) {
        nvtxEventAttributes_t attr {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = argb;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx(&attr);
    }

    ~ScopedNvtxRange() {
        nvtxRangePop();
    }
};

}  // namespace gpu_mvr::profile

#define GPU_MVR_NVTX_SCOPE(name, color) \
    ::gpu_mvr::profile::ScopedNvtxRange gpu_mvr_nvtx_scope_##__LINE__{(name), (color)}
#else
#define GPU_MVR_NVTX_SCOPE(name, color) do {} while (0)
#endif
