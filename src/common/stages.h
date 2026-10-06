#pragma once
// Pipeline stage interfaces.
//  * A "Backend" is a table of the data-parallel stages. SEQ_BACKEND and OMP_BACKEND
//    implement the SAME stages; they differ only in how rows/pixels are scheduled.
//  * Everything else (Otsu, LUT, connected components, features, classifier) is shared.
#include <cstdint>
#include <string>
#include <vector>
#include "common/image.h"

struct Sums { uint64_t n = 0, sum = 0, sumsq = 0, edge = 0, mask_on = 0; };

struct Backend {
    const char* name;
    Image (*resize)(const Image&, int size);
    Image (*gaussian5)(const Image&);
    void  (*histogram)(const Image&, uint32_t* hist256);
    Image (*apply_lut)(const Image&, const uint8_t* lut256);
    Image (*threshold)(const Image&, int t);
    Image (*open3)(const Image&);                       // 3x3 erosion then dilation
    Sums  (*sums)(const Image& gray, const Image& mask);
};
extern const Backend SEQ_BACKEND;
#ifdef USE_OPENMP
extern const Backend OMP_BACKEND;
#endif

// ---- shared (sequential, cheap) helpers: stages_common.cpp ----
struct ResizeMap { std::vector<int> i0, i1; std::vector<uint32_t> f; };
ResizeMap build_resize_map(int src, int dst);           // 16.16 fixed-point bilinear taps
void make_equalize_lut(const uint32_t* hist, long total, uint8_t* lut);
int  otsu_threshold(const uint32_t* hist, long total);
std::vector<Roi> connected_components(const Image& mask, int min_area);   // 8-connectivity

constexpr int NFEAT = 6;
extern const char* FEATURE_NAMES[NFEAT];
struct Features { double v[NFEAT]; };
Features compute_features(const Sums& s, const std::vector<Roi>& rois, int w, int h);

struct Model { double w[NFEAT]; double b; };
Model default_model();
bool  load_model(const std::string& path, Model& m);    // NFEAT weights then bias
double model_score(const Model& m, const Features& f);  // > 0  => ABNORMAL
