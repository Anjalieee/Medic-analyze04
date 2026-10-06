#pragma once
// Row-level kernels shared by the sequential and OpenMP backends.
// Because both backends call the SAME kernels, any difference in output can only come
// from scheduling (race conditions, bad reductions), which is exactly what the
// correctness tests are designed to catch.
#include <algorithm>
#include <cstdlib>
#include "common/image.h"
#include "common/stages.h"

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline constexpr int GAUSS5[5] = {1, 4, 6, 4, 1};       // separable [1 4 6 4 1]/16, integer maths

inline void resize_row(const Image& s, Image& o, int y, const ResizeMap& mx, const ResizeMap& my) {
    const uint8_t* r0 = s.row(my.i0[y]);
    const uint8_t* r1 = s.row(my.i1[y]);
    const uint64_t fy = my.f[y];
    uint8_t* out = o.row(y);
    for (int x = 0; x < o.w; ++x) {
        const uint64_t fx = mx.f[x];
        const uint64_t top = r0[mx.i0[x]] * (65536 - fx) + r0[mx.i1[x]] * fx;
        const uint64_t bot = r1[mx.i0[x]] * (65536 - fx) + r1[mx.i1[x]] * fx;
        out[x] = static_cast<uint8_t>((top * (65536 - fy) + bot * fy + (1ULL << 31)) >> 32);
    }
}

inline void gauss_h_row(const Image& s, Image& t, int y) {
    const uint8_t* r = s.row(y);
    uint8_t* o = t.row(y);
    const int w = s.w;
    for (int x = 0; x < w; ++x) {
        int sum = 0;
        for (int k = -2; k <= 2; ++k) sum += GAUSS5[k + 2] * r[clampi(x + k, 0, w - 1)];
        o[x] = static_cast<uint8_t>((sum + 8) >> 4);
    }
}

inline void gauss_v_row(const Image& t, Image& o, int y) {
    const uint8_t* rows[5];
    for (int k = -2; k <= 2; ++k) rows[k + 2] = t.row(clampi(y + k, 0, t.h - 1));
    uint8_t* out = o.row(y);
    for (int x = 0; x < t.w; ++x) {
        int sum = 0;
        for (int k = 0; k < 5; ++k) sum += GAUSS5[k] * rows[k][x];
        out[x] = static_cast<uint8_t>((sum + 8) >> 4);
    }
}

inline void erode_row(const Image& s, Image& o, int y) {      // border pixels become 0
    const int w = s.w, h = s.h;
    uint8_t* out = o.row(y);
    std::fill(out, out + w, 0);
    if (y == 0 || y == h - 1) return;
    const uint8_t *a = s.row(y - 1), *b = s.row(y), *c = s.row(y + 1);
    for (int x = 1; x < w - 1; ++x)
        out[x] = (a[x-1] && a[x] && a[x+1] && b[x-1] && b[x] && b[x+1] &&
                  c[x-1] && c[x] && c[x+1]) ? 255 : 0;
}

inline void dilate_row(const Image& s, Image& o, int y) {
    const int w = s.w, h = s.h;
    uint8_t* out = o.row(y);
    for (int x = 0; x < w; ++x) {
        uint8_t v = 0;
        for (int dy = -1; dy <= 1 && !v; ++dy) {
            const int yy = y + dy;
            if (yy < 0 || yy >= h) continue;
            const uint8_t* r = s.row(yy);
            for (int dx = -1; dx <= 1; ++dx) {
                const int xx = x + dx;
                if (xx >= 0 && xx < w && r[xx]) { v = 255; break; }
            }
        }
        out[x] = v;
    }
}

// Accumulates intensity sums, mask count and Sobel edge strength (|gx|+|gy|) for one row.
// All integer arithmetic, so the final sums are identical for ANY summation order.
inline void sums_row(const Image& g, const Image& m, int y, Sums& a) {
    const int w = g.w;
    const uint8_t* r = g.row(y);
    const uint8_t* mr = m.row(y);
    for (int x = 0; x < w; ++x) {
        a.sum += r[x];
        a.sumsq += static_cast<uint64_t>(r[x]) * r[x];
        if (mr[x]) a.mask_on++;
    }
    if (y >= 1 && y < g.h - 1) {
        const uint8_t *u = g.row(y - 1), *c = r, *d = g.row(y + 1);
        for (int x = 1; x < w - 1; ++x) {
            const int gx = (u[x+1] + 2 * c[x+1] + d[x+1]) - (u[x-1] + 2 * c[x-1] + d[x-1]);
            const int gy = (d[x-1] + 2 * d[x] + d[x+1]) - (u[x-1] + 2 * u[x] + u[x+1]);
            a.edge += static_cast<uint64_t>(std::abs(gx) + std::abs(gy));
        }
    }
}
