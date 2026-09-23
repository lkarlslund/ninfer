#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* path = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (path == nullptr || *path == '\0') { return 77; }
    try {
        ninfer::artifact::Reader reader{std::filesystem::path(path)};
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(binder);
        if (plan.bindings.ple_mapping.size() != 320001536ULL * 160) {
            throw std::runtime_error("PLE table mapping has the wrong extent");
        }
        if (plan.materialization.device_objects.size() != 1260) {
            throw std::runtime_error("MTP0/Vision-off load plan uploaded optional tensors");
        }
        ninfer::artifact::Binder mtp_binder(reader);
        const auto mtp_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            mtp_binder, {.speculative = ninfer::SpeculativeBackend::Mtp});
        if (mtp_plan.materialization.device_objects.size() != 1291 ||
            mtp_plan.materialization.device_capacity_bytes <=
                plan.materialization.device_capacity_bytes) {
            throw std::runtime_error("MTP3 load plan did not upload exactly the draft tensors");
        }
        ninfer::artifact::Binder full_binder(reader);
        const auto full_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            full_binder, {.vision = true, .speculative = ninfer::SpeculativeBackend::Mtp});
        if (full_plan.materialization.device_objects.size() != 1624 ||
            full_plan.materialization.device_capacity_bytes <=
                mtp_plan.materialization.device_capacity_bytes) {
            throw std::runtime_error("Vision load plan did not upload exactly the Vision tensors");
        }
        std::cout << "device=" << plan.materialization.device_capacity_bytes
                  << " mtp_device=" << mtp_plan.materialization.device_capacity_bytes
                  << " full_device=" << full_plan.materialization.device_capacity_bytes
                  << " file_backed="
                  << plan.bindings.ple_mapping.size()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
