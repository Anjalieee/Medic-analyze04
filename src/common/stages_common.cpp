// Shared, inherently sequential or very cheap stages (identical in every build).
#include "common/stages.h"
#include <algorithm>
#include <cmath>
#include <fstream>

ResizeMap build_resize_map(int src, int dst) {
    ResizeMap m;
    m.i0.resize(dst); m.i1.resize(dst); m.f.resize(dst);
    for (int i = 0; i < dst; ++i) {
        // source coordinate of pixel centre, 16.16 fixed point: (i+0.5)*src/dst - 0.5
        int64_t p = (static_cast<int64_t>(2 * i + 1) * src * 65536) / (2 * static_cast<int64_t>(dst)) - 32768;
        if (p < 0) p = 0;
        int a = static_cast<int>(p >> 16);
        if (a > src - 1) a = src - 1;
        m.i0[i] = a;
        m.i1[i] = std::min(a + 1, src - 1);
        m.f[i] = static_cast<uint32_t>(p & 0xFFFF);
    }
    return m;
}

void make_equalize_lut(const uint32_t* hist, long total, uint8_t* lut) {
    uint64_t cdf = 0, cdf_min = 0;
    uint64_t c[256];
    for (int i = 0; i < 256; ++i) { cdf += hist[i]; c[i] = cdf; if (!cdf_min && cdf) cdf_min = cdf; }
    const uint64_t denom = static_cast<uint64_t>(total) - cdf_min;
    for (int i = 0; i < 256; ++i) {
        if (denom == 0) lut[i] = static_cast<uint8_t>(i);            // flat image: identity
        else lut[i] = static_cast<uint8_t>(c[i] <= cdf_min ? 0 : ((c[i] - cdf_min) * 255 + denom / 2) / denom);
    }
}

int otsu_threshold(const uint32_t* hist, long total) {
    double sum = 0;
    for (int i = 0; i < 256; ++i) sum += i * static_cast<double>(hist[i]);
    double sumB = 0, wB = 0, best = -1;
    int t = 0;
    for (int i = 0; i < 256; ++i) {
        wB += hist[i];
        if (wB == 0) continue;
        const double wF = static_cast<double>(total) - wB;
        if (wF == 0) break;
        sumB += i * static_cast<double>(hist[i]);
        const double mB = sumB / wB, mF = (sum - sumB) / wF;
        const double var = wB * wF * (mB - mF) * (mB - mF);
        if (var > best) { best = var; t = i; }
    }
    return t;
}

// Iterative flood fill, raster-scan order => deterministic ROI ordering.
std::vector<Roi> connected_components(const Image& m, int min_area) {
    const int w = m.w, h = m.h;
    std::vector<uint8_t> vis(m.d);                       // non-zero = foreground, not yet visited
    std::vector<int> stack;
    std::vector<Roi> out;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = static_cast<size_t>(y) * w + x;
            if (!vis[i]) continue;
            vis[i] = 0;
            stack.push_back(static_cast<int>(i));
            Roi r{x, y, x, y, 0};
            while (!stack.empty()) {
                const int c = stack.back(); stack.pop_back();
                const int cx = c % w, cy = c / w;
                r.area++;
                r.x0 = std::min(r.x0, cx); r.x1 = std::max(r.x1, cx);
                r.y0 = std::min(r.y0, cy); r.y1 = std::max(r.y1, cy);
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        const size_t ni = static_cast<size_t>(ny) * w + nx;
                        if (vis[ni]) { vis[ni] = 0; stack.push_back(static_cast<int>(ni)); }
                    }
            }
            if (r.area >= min_area) out.push_back(r);
        }
    return out;
}

const char* FEATURE_NAMES[NFEAT] = {"bright_frac", "largest_roi_frac", "mean", "std", "edge_density", "roi_count"};

Features compute_features(const Sums& s, const std::vector<Roi>& rois, int w, int h) {
    Features f{};
    const double n = static_cast<double>(s.n);
    const double mean = s.sum / n;
    const double var = std::max(0.0, s.sumsq / n - mean * mean);
    int largest = 0;
    for (const Roi& r : rois) largest = std::max(largest, r.area);
    const double interior = (w > 2 && h > 2) ? double(w - 2) * (h - 2) : 1.0;
    f.v[0] = s.mask_on / n;                              // fraction of pixels flagged bright
    f.v[1] = largest / n;                                // largest ROI as fraction of image
    f.v[2] = mean / 255.0;
    f.v[3] = std::sqrt(var) / 255.0;
    f.v[4] = s.edge / (interior * 2040.0);               // 2040 = max |gx|+|gy| for 8-bit Sobel
    f.v[5] = std::min<size_t>(rois.size(), 10) / 10.0;
    return f;
}

// PLACEHOLDER weights so the pipeline runs end-to-end. They are NOT trained.
// Step 6 fits real weights from the exported features and passes them with --model.
Model default_model() {
    return Model{{1.5, 2.5, 3.0, 1.0, -1.0, 0.5}, -3.0};
}

bool load_model(const std::string& path, Model& m) {
    std::ifstream f(path);
    if (!f) return false;
    for (int i = 0; i < NFEAT; ++i) if (!(f >> m.w[i])) return false;
    return static_cast<bool>(f >> m.b);
}

double model_score(const Model& m, const Features& f) {
    double s = m.b;
    for (int i = 0; i < NFEAT; ++i) s += m.w[i] * f.v[i];
    return s;
}
