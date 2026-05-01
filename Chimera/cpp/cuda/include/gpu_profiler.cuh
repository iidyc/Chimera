#pragma once

#include <cstdint>

#ifdef CHIMERA_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>

namespace Chimera::profile {

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

}  // namespace Chimera::profile

#define CHIMERA_NVTX_SCOPE(name, color) \
    ::Chimera::profile::ScopedNvtxRange chimera_nvtx_scope_##__LINE__{(name), (color)}
#else
#define CHIMERA_NVTX_SCOPE(name, color) do {} while (0)
#endif
