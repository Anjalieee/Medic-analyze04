// Medical image analysis pipeline - driver for the sequential and OpenMP versions.
//   mia_seq --images DIR [options]                      (sequential baseline, no OpenMP)
//   mia_omp --mode omp_images|omp_pixels --threads N ...
// Modes:
//   seq         : one thread, one image at a time
//   omp_images  : parallel over IMAGES (each thread runs the sequential stages on its own image)
//   omp_pixels  : images one at a time, rows/pixels INSIDE each stage run in parallel
#include <sys/resource.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "common/metrics.h"
#include "common/pipeline.h"
#include "common/timer.h"

struct Item { std::string file; int label; };

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
        out.push_back(cell);
    }
    return out;
}

static bool read_labels(const std::string& csv, const std::string& split, std::vector<Item>& items) {
    std::ifstream f(csv);
    if (!f) return false;
    std::string line;
    std::getline(f, line);
    const auto hdr = split_csv(line);
    int ci_file = -1, ci_label = -1, ci_split = -1;
    for (size_t i = 0; i < hdr.size(); ++i) {
        if (hdr[i] == "filename") ci_file = static_cast<int>(i);
        if (hdr[i] == "label")    ci_label = static_cast<int>(i);
        if (hdr[i] == "split")    ci_split = static_cast<int>(i);
    }
    if (ci_file < 0 || ci_label < 0) return false;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto c = split_csv(line);
        if (static_cast<int>(c.size()) <= std::max(ci_file, ci_label)) continue;
        if (!split.empty() && ci_split >= 0 && c[ci_split] != split) continue;
        items.push_back({c[ci_file], std::atoi(c[ci_label].c_str())});
    }
    return true;
}

static void usage() {
    std::puts("usage: mia --images DIR [--labels CSV] [--split train|val|test] [--mode seq|omp_images|omp_pixels]\n"
              "           [--threads N] [--size 512] [--limit N] [--out results.csv] [--summary summary.csv]\n"
              "           [--save-dir DIR] [--model model.txt] [--no-hash]");
}

int main(int argc, char** argv) {
    std::string images, labels, split, mode = "seq", out_csv, summary, save_dir, model_path;
    int threads = 1, size = 512;
    long limit = -1;
    bool hash = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : (usage(), exit(1), ""); };
        if (a == "--images") images = next();
        else if (a == "--labels") labels = next();
        else if (a == "--split") split = next();
        else if (a == "--mode") mode = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--size") size = std::atoi(next().c_str());
        else if (a == "--limit") limit = std::atol(next().c_str());
        else if (a == "--out") out_csv = next();
        else if (a == "--summary") summary = next();
        else if (a == "--save-dir") save_dir = next();
        else if (a == "--model") model_path = next();
        else if (a == "--no-hash") hash = false;
        else { usage(); return 1; }
    }
    if (images.empty()) { usage(); return 1; }
    if (labels.empty()) labels = images + "/labels.csv";

    const Backend* be = &SEQ_BACKEND;      // backend used INSIDE one image
    bool par_images = false;
    if (mode == "seq") {
        threads = 1;
    } else if (mode == "omp_images" || mode == "omp_pixels") {
#ifdef USE_OPENMP
        if (threads < 1) threads = 1;
        omp_set_num_threads(threads);
        if (mode == "omp_pixels") be = &OMP_BACKEND; else par_images = true;
#else
        std::fprintf(stderr, "This binary was built WITHOUT OpenMP; use mia_omp for mode '%s'.\n", mode.c_str());
        return 1;
#endif
    } else { usage(); return 1; }

    std::vector<Item> items;
    if (!read_labels(labels, split, items)) { std::fprintf(stderr, "Cannot read labels: %s\n", labels.c_str()); return 1; }
    if (limit > 0 && static_cast<size_t>(limit) < items.size()) items.resize(limit);
    if (items.empty()) { std::fprintf(stderr, "No images selected.\n"); return 1; }

    PipelineOptions opt;
    opt.size = size;
    opt.hash = hash;
    if (!model_path.empty() && !load_model(model_path, opt.model)) { std::fprintf(stderr, "Bad model file\n"); return 1; }

    const long N = static_cast<long>(items.size());
    std::vector<Result> res(N);                       // one slot per image => no shared writes, fixed order

    const double t0 = now_ms();
    if (par_images) {
#ifdef USE_OPENMP
        // Images vary in size/cost -> dynamic scheduling balances the load.
        #pragma omp parallel for schedule(dynamic, 1)
        for (long i = 0; i < N; ++i)
            res[i] = process_image(images + "/" + items[i].file, items[i].file, items[i].label, *be, opt, save_dir);
#endif
    } else {
        for (long i = 0; i < N; ++i)
            res[i] = process_image(images + "/" + items[i].file, items[i].file, items[i].label, *be, opt, save_dir);
    }
    const double wall_s = (now_ms() - t0) / 1000.0;

    // ---- aggregate ----
    Confusion cm;
    double stage_sum[NSTAGES] = {0};
    long failed = 0;
    for (const Result& r : res) {
        if (!r.ok) { ++failed; continue; }
        if (r.label == 0 || r.label == 1) cm.add(r.label, r.pred);   // label -1 = unknown, not scored
        for (int s = 0; s < NSTAGES; ++s) stage_sum[s] += r.t_ms[s];
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    const double peak_mb = ru.ru_maxrss / (1024.0 * 1024.0);   // macOS reports BYTES
#else
    const double peak_mb = ru.ru_maxrss / 1024.0;              // Linux reports KILOBYTES
#endif

    // ---- per-image CSV ----
    if (!out_csv.empty()) {
        std::ofstream o(out_csv);
        o << "file,label,pred,score";
        for (int k = 0; k < NFEAT; ++k) o << "," << FEATURE_NAMES[k];
        o << ",n_rois,h_resized,h_denoised,h_enhanced,h_mask,rois";
        for (int s = 0; s < NSTAGES; ++s) o << ",t_" << STAGE_NAMES[s] << "_ms";
        o << "\n";
        char buf[128];
        for (const Result& r : res) {
            if (!r.ok) continue;
            o << r.file << "," << r.label << "," << r.pred;
            std::snprintf(buf, sizeof buf, ",%.9f", r.score); o << buf;
            for (int k = 0; k < NFEAT; ++k) { std::snprintf(buf, sizeof buf, ",%.9f", r.feat.v[k]); o << buf; }
            o << "," << r.rois.size();
            std::snprintf(buf, sizeof buf, ",%016llx,%016llx,%016llx,%016llx",
                          (unsigned long long)r.h_resized, (unsigned long long)r.h_denoised,
                          (unsigned long long)r.h_enhanced, (unsigned long long)r.h_mask);
            o << buf << ",";
            for (size_t k = 0; k < r.rois.size(); ++k) {
                const Roi& q = r.rois[k];
                o << (k ? "|" : "") << q.x0 << ":" << q.y0 << ":" << q.x1 << ":" << q.y1 << ":" << q.area;
            }
            for (int s = 0; s < NSTAGES; ++s) { std::snprintf(buf, sizeof buf, ",%.4f", r.t_ms[s]); o << buf; }
            o << "\n";
        }
    }

    // ---- one-line summary row (appended) for benchmark scripts ----
    if (!summary.empty()) {
        bool fresh = false;
        { std::ifstream t(summary); fresh = !t.good() || t.peek() == std::ifstream::traits_type::eof(); }
        std::ofstream o(summary, std::ios::app);
        if (fresh) {
            o << "mode,threads,images,size,wall_s,img_per_s,peak_mb,accuracy,precision,recall,f1";
            for (int s = 0; s < NSTAGES; ++s) o << ",sum_" << STAGE_NAMES[s] << "_ms";
            o << "\n";
        }
        o << mode << "," << threads << "," << (N - failed) << "," << size << "," << wall_s << ","
          << (N - failed) / wall_s << "," << peak_mb << "," << cm.accuracy() << "," << cm.precision() << ","
          << cm.recall() << "," << cm.f1();
        for (int s = 0; s < NSTAGES; ++s) o << "," << stage_sum[s];
        o << "\n";
    }

    // ---- console report ----
    std::printf("mode=%s threads=%d images=%ld (failed %ld) size=%d\n", mode.c_str(), threads, N - failed, failed, size);
    std::printf("wall time : %.3f s   (%.1f images/s)\n", wall_s, (N - failed) / wall_s);
    std::printf("peak RSS  : %.1f MB\n", peak_mb);
    double tot = 0;
    for (int s = 0; s < NSTAGES; ++s) tot += stage_sum[s];
    std::printf("stage time summed over images (CPU-ms%s):\n", par_images ? ", summed across threads" : "");
    for (int s = 0; s < NSTAGES; ++s)
        std::printf("  %-9s %10.1f ms  %5.1f%%\n", STAGE_NAMES[s], stage_sum[s], tot > 0 ? 100.0 * stage_sum[s] / tot : 0.0);
    std::printf("classification (%s):\n", model_path.empty() ? "default weights are untrained placeholders" : "trained model");
    std::printf("  TP=%ld FP=%ld TN=%ld FN=%ld  acc=%.3f prec=%.3f rec=%.3f f1=%.3f\n",
                cm.tp, cm.fp, cm.tn, cm.fn, cm.accuracy(), cm.precision(), cm.recall(), cm.f1());
    return failed ? 2 : 0;
}
