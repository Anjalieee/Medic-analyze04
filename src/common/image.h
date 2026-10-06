#pragma once
// Basic image container and I/O declarations.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Image {                      // 8-bit grayscale image, row-major
    int w = 0, h = 0;
    std::vector<uint8_t> d;
    Image() = default;
    Image(int w_, int h_) : w(w_), h(h_), d(static_cast<size_t>(w_) * h_, 0) {}
    uint8_t* row(int y) { return d.data() + static_cast<size_t>(y) * w; }
    const uint8_t* row(int y) const { return d.data() + static_cast<size_t>(y) * w; }
    size_t size() const { return d.size(); }
};

struct Roi { int x0, y0, x1, y1, area; };   // inclusive bounding box + pixel count

bool load_gray(const std::string& path, Image& out);                 // any format -> 8-bit gray
bool save_gray_png(const std::string& path, const Image& img);
bool save_overlay_png(const std::string& path, const Image& gray, const std::vector<Roi>& rois);
uint64_t fnv1a(const uint8_t* p, size_t n);                           // 64-bit FNV-1a hash
