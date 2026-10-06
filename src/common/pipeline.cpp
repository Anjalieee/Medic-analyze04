#include "common/pipeline.h"
#include "common/timer.h"
#include <algorithm>

const char* STAGE_NAMES[NSTAGES] = {"load", "resize", "denoise", "enhance", "segment", "roi", "features"};

Result process_image(const std::string& path, const std::string& name, int label,
                     const Backend& be, const PipelineOptions& opt, const std::string& save_dir) {
    Result r;
    r.file = name;
    r.label = label;
    double t = now_ms();
    auto lap = [&](Stage s) { double n = now_ms(); r.t_ms[s] = n - t; t = n; };

    Image g;
    if (!load_gray(path, g)) return r;                                   // r.ok stays false
    r.src_w = g.w; r.src_h = g.h;
    lap(ST_LOAD);

    Image rs = be.resize(g, opt.size);                                    lap(ST_RESIZE);
    Image bl = be.gaussian5(rs);                                          lap(ST_DENOISE);

    const long n = static_cast<long>(bl.size());
    uint32_t hist[256];
    uint8_t lut[256];
    be.histogram(bl, hist);
    make_equalize_lut(hist, n, lut);
    Image eq = be.apply_lut(bl, lut);                                     lap(ST_ENHANCE);

    // Otsu runs on the DENOISED image's histogram (already computed above), not on the
    // equalized one: equalization flattens the histogram and destroys the two-peak shape
    // that Otsu relies on (found via the known-answer test T5).
    const int th = otsu_threshold(hist, n);
    Image mk = be.open3(be.threshold(bl, th));                            lap(ST_SEGMENT);

    const int min_area = std::max(50, static_cast<int>(n / 500));         // ignore specks (<0.2%)
    r.rois = connected_components(mk, min_area);                          lap(ST_ROI);

    const Sums s = be.sums(eq, mk);
    r.feat = compute_features(s, r.rois, eq.w, eq.h);
    r.score = model_score(opt.model, r.feat);
    r.pred = r.score > 0 ? 1 : 0;
    lap(ST_FEATURES);
    r.ok = true;

    // ---- everything below is NOT included in the stage timings ----
    if (opt.hash) {
        r.h_resized  = fnv1a(rs.d.data(), rs.size());
        r.h_denoised = fnv1a(bl.d.data(), bl.size());
        r.h_enhanced = fnv1a(eq.d.data(), eq.size());
        r.h_mask     = fnv1a(mk.d.data(), mk.size());
    }
    if (!save_dir.empty()) {
        const std::string base = save_dir + "/" + name;
        save_gray_png(base + "_enh.png", eq);
        save_gray_png(base + "_mask.png", mk);
        save_overlay_png(base + "_overlay.png", eq, r.rois);
    }
    return r;
}
