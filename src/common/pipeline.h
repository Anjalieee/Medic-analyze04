#pragma once
#include <string>
#include <vector>
#include "common/stages.h"

enum Stage { ST_LOAD, ST_RESIZE, ST_DENOISE, ST_ENHANCE, ST_SEGMENT, ST_ROI, ST_FEATURES, NSTAGES };
extern const char* STAGE_NAMES[NSTAGES];

struct PipelineOptions {
    int  size = 512;          // images are resized to size x size
    bool hash = true;         // compute checksums of intermediates (for correctness checks)
    Model model = default_model();
};

struct Result {
    bool ok = false;
    std::string file;
    int label = -1, pred = -1, src_w = 0, src_h = 0;
    double score = 0;
    Features feat{};
    std::vector<Roi> rois;
    uint64_t h_resized = 0, h_denoised = 0, h_enhanced = 0, h_mask = 0;
    double t_ms[NSTAGES] = {0};
};

// Runs the full pipeline for one image with the given backend.
// If save_dir is non-empty, writes <name>_enh.png, <name>_mask.png, <name>_overlay.png.
Result process_image(const std::string& path, const std::string& name, int label,
                     const Backend& be, const PipelineOptions& opt, const std::string& save_dir);
