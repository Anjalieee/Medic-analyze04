// OpenMP backend: SAME row kernels as the sequential backend, parallelised over rows/pixels.
// Each parallel loop writes disjoint output rows (no races); reductions are explicit.
#ifdef USE_OPENMP
#include <omp.h>
#include "common/kernels.h"

static Image resize_omp(const Image& s, int N) {
    Image o(N, N);
    const ResizeMap mx = build_resize_map(s.w, N), my = build_resize_map(s.h, N);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < N; ++y) resize_row(s, o, y, mx, my);
    return o;
}

static Image gaussian5_omp(const Image& s) {
    Image t(s.w, s.h), o(s.w, s.h);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < s.h; ++y) gauss_h_row(s, t, y);
    // implicit barrier above: the vertical pass needs ALL rows of t finished
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < s.h; ++y) gauss_v_row(t, o, y);
    return o;
}

// Histogram: array reduction => every thread fills a PRIVATE copy, merged at the end.
// (A shared hist[v]++ here would be a data race.)
static void histogram_omp(const Image& s, uint32_t* hist) {
    uint32_t local[256] = {0};
    const long n = static_cast<long>(s.size());
    #pragma omp parallel for schedule(static) reduction(+ : local[:256])
    for (long i = 0; i < n; ++i) local[s.d[i]]++;
    std::copy(local, local + 256, hist);
}

static Image apply_lut_omp(const Image& s, const uint8_t* lut) {
    Image o(s.w, s.h);
    const long n = static_cast<long>(s.size());
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) o.d[i] = lut[s.d[i]];
    return o;
}

static Image threshold_omp(const Image& s, int t) {
    Image o(s.w, s.h);
    const long n = static_cast<long>(s.size());
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) o.d[i] = s.d[i] > t ? 255 : 0;
    return o;
}

static Image open3_omp(const Image& s) {
    Image e(s.w, s.h), o(s.w, s.h);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < s.h; ++y) erode_row(s, e, y);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < s.h; ++y) dilate_row(e, o, y);
    return o;
}

static Sums sums_omp(const Image& g, const Image& m) {
    uint64_t sum = 0, sumsq = 0, edge = 0, on = 0;
    #pragma omp parallel for schedule(static) reduction(+ : sum, sumsq, edge, on)
    for (int y = 0; y < g.h; ++y) {
        Sums a;
        sums_row(g, m, y, a);
        sum += a.sum; sumsq += a.sumsq; edge += a.edge; on += a.mask_on;
    }
    Sums r;
    r.n = g.size(); r.sum = sum; r.sumsq = sumsq; r.edge = edge; r.mask_on = on;
    return r;
}

const Backend OMP_BACKEND = {"omp", resize_omp, gaussian5_omp, histogram_omp,
                             apply_lut_omp, threshold_omp, open3_omp, sums_omp};
#endif
