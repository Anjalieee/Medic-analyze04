#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "common/image.h"
#include <cstring>

bool load_gray(const std::string& path, Image& out) {
    int w = 0, h = 0, c = 0;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &c, 1);   // force 1 channel (grayscale)
    if (!p) return false;
    out = Image(w, h);
    std::memcpy(out.d.data(), p, out.size());
    stbi_image_free(p);
    return true;
}

bool save_gray_png(const std::string& path, const Image& img) {
    return stbi_write_png(path.c_str(), img.w, img.h, 1, img.d.data(), img.w) != 0;
}

bool save_overlay_png(const std::string& path, const Image& g, const std::vector<Roi>& rois) {
    std::vector<uint8_t> rgb(g.size() * 3);
    for (size_t i = 0; i < g.size(); ++i) rgb[3*i] = rgb[3*i+1] = rgb[3*i+2] = g.d[i];
    auto put = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= g.w || y >= g.h) return;
        uint8_t* p = &rgb[(static_cast<size_t>(y) * g.w + x) * 3];
        p[0] = 255; p[1] = 0; p[2] = 0;
    };
    for (const Roi& r : rois)
        for (int t = 0; t < 2; ++t) {                        // 2-pixel red box
            for (int x = r.x0; x <= r.x1; ++x) { put(x, r.y0 + t); put(x, r.y1 - t); }
            for (int y = r.y0; y <= r.y1; ++y) { put(r.x0 + t, y); put(r.x1 - t, y); }
        }
    return stbi_write_png(path.c_str(), g.w, g.h, 3, rgb.data(), g.w * 3) != 0;
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
