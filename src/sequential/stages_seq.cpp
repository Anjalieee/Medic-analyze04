// SEQUENTIAL backend: plain loops, no OpenMP anywhere.
#include "common/kernels.h"

static Image resize_seq(const Image& s, int N) {
    Image o(N, N);
    const ResizeMap mx = build_resize_map(s.w, N), my = build_resize_map(s.h, N);
    for (int y = 0; y < N; ++y) resize_row(s, o, y, mx, my);
    return o;
}

static Image gaussian5_seq(const Image& s) {
    Image t(s.w, s.h), o(s.w, s.h);
    for (int y = 0; y < s.h; ++y) gauss_h_row(s, t, y);
    for (int y = 0; y < s.h; ++y) gauss_v_row(t, o, y);
    return o;
}

static void histogram_seq(const Image& s, uint32_t* hist) {
    std::fill(hist, hist + 256, 0u);
    for (size_t i = 0; i < s.size(); ++i) hist[s.d[i]]++;
}

static Image apply_lut_seq(const Image& s, const uint8_t* lut) {
    Image o(s.w, s.h);
    for (size_t i = 0; i < s.size(); ++i) o.d[i] = lut[s.d[i]];
    return o;
}

static Image threshold_seq(const Image& s, int t) {
    Image o(s.w, s.h);
    for (size_t i = 0; i < s.size(); ++i) o.d[i] = s.d[i] > t ? 255 : 0;
    return o;
}

static Image open3_seq(const Image& s) {
    Image e(s.w, s.h), o(s.w, s.h);
    for (int y = 0; y < s.h; ++y) erode_row(s, e, y);
    for (int y = 0; y < s.h; ++y) dilate_row(e, o, y);
    return o;
}

static Sums sums_seq(const Image& g, const Image& m) {
    Sums a;
    a.n = g.size();
    for (int y = 0; y < g.h; ++y) sums_row(g, m, y, a);
    return a;
}

const Backend SEQ_BACKEND = {"seq", resize_seq, gaussian5_seq, histogram_seq,
                             apply_lut_seq, threshold_seq, open3_seq, sums_seq};
