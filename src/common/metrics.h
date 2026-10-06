#pragma once
// Binary classification metrics. Positive class = ABNORMAL (label 1).
struct Confusion {
    long tp = 0, fp = 0, tn = 0, fn = 0;
    void add(int label, int pred) {
        if (label == 1) (pred == 1 ? tp : fn)++;
        else            (pred == 1 ? fp : tn)++;
    }
    long total() const { return tp + fp + tn + fn; }
    double accuracy()  const { return total() ? double(tp + tn) / total() : 0.0; }
    double precision() const { return (tp + fp) ? double(tp) / (tp + fp) : 0.0; }
    double recall()    const { return (tp + fn) ? double(tp) / (tp + fn) : 0.0; }
    double f1() const {
        double p = precision(), r = recall();
        return (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
    }
};
