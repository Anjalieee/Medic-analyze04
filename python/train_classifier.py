#!/usr/bin/env python3
"""
Step 6 - train the NORMAL/ABNORMAL classifier on features produced by the C++ pipeline.

Flow
 1. Run the C++ pipeline on the train / val / test splits -> per-image feature CSVs
 2. Fit L2-regularised, class-balanced logistic regression on TRAIN (numpy only)
 3. Choose the decision threshold on VAL (never on test)
 4. Fold standardisation + threshold into raw weights -> models/model.txt (read by --model)
 5. Evaluate on TEST by running the real C++ binary with that model (end-to-end check)
 6. Write results/classifier_metrics.json

Usage:  python3 python/train_classifier.py --images data/raw --labels data/labels.csv
"""
import argparse, csv, json, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin" / "mia_omp"
FEATS = ["bright_frac", "largest_roi_frac", "mean", "std", "edge_density", "roi_count"]

def extract(images, labels, split, out, threads):
    cmd = [str(BIN), "--images", images, "--labels", labels, "--split", split, "--mode", "omp_images",
           "--threads", str(threads), "--no-hash", "--out", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATS] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    pred = np.array([int(r["pred"]) for r in rows])
    return X, y, pred, [r["file"] for r in rows]

def fit_logreg(X, y, l2=1.0, iters=50):
    """Newton / IRLS with class-balanced weights. Returns (theta, b0) on standardised features."""
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    cw = np.where(y == 1, n / (2.0 * y.sum()), n / (2.0 * (n - y.sum())))
    th = np.zeros(d + 1)
    R = l2 * np.eye(d + 1); R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ th))
        g = A.T @ (cw * (p - y)) + R @ th
        H = A.T @ (A * (cw * p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H + 1e-9 * np.eye(d + 1), g)
        th -= step
        if np.abs(step).max() < 1e-8: break
    return th[1:], th[0]

def metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                specificity=spec, f1=(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                balanced_accuracy=(rec + spec) / 2)

def pick_threshold(logit, y, criterion):
    best, bt = -1, 0.0
    for t in np.unique(np.quantile(logit, np.linspace(0.01, 0.99, 197))):
        m = metrics(y, (logit > t).astype(int))
        s = {"youden": m["balanced_accuracy"], "f1": m["f1"]}[criterion]
        if s > best: best, bt = s, t
    return bt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--criterion", choices=["youden", "f1"], default="youden",
                    help="threshold selection on val (youden = balanced accuracy, avoids the all-abnormal trap)")
    a = ap.parse_args()
    out = ROOT / "results" / "features"; out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)

    D = {}
    for s in ("train", "val", "test"):
        extract(a.images, a.labels, s, out / f"{s}.csv", a.threads)
        D[s] = load(out / f"{s}.csv")
        print(f"{s:5s}: {len(D[s][1])} images, {D[s][1].mean():.1%} abnormal")

    Xtr, ytr = D["train"][0], D["train"][1]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    th, b0 = fit_logreg((Xtr - mu) / sd, ytr, a.l2)
    z = lambda X: b0 + ((X - mu) / sd) @ th
    t = pick_threshold(z(D["val"][0]), D["val"][1], a.criterion)

    w_raw = th / sd                                     # fold standardisation into the weights
    b_raw = b0 - float((th * mu / sd).sum()) - t        # ... and the threshold into the bias
    model = ROOT / "models" / "model.txt"
    model.write_text(" ".join(f"{v:.12g}" for v in w_raw) + f"\n{b_raw:.12g}\n")
    print("\nlearned weights (raw feature space):")
    for n, w in zip(FEATS, w_raw): print(f"  {n:18s} {w:+.4f}")
    print(f"  {'bias':18s} {b_raw:+.4f}\n-> {model}")

    # Evaluate through the real C++ binary using the saved model (end-to-end verification)
    report = {"criterion": a.criterion, "weights": dict(zip(FEATS, w_raw.tolist())), "bias": b_raw}
    for s in ("val", "test"):
        extract_cmd = [str(BIN), "--images", a.images, "--labels", a.labels, "--split", s, "--mode", "omp_images",
                       "--threads", str(a.threads), "--no-hash", "--model", str(model), "--out", str(out / f"{s}_pred.csv")]
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL)
        X, y, cpp_pred, _ = load(out / f"{s}_pred.csv")
        np_pred = (z(X) > t).astype(int)
        agree = float((np_pred == cpp_pred).mean())
        m = metrics(y, cpp_pred)
        m["python_vs_cpp_agreement"] = agree
        m["baseline_all_abnormal_accuracy"] = float(y.mean())
        report[s] = m
        print(f"\n[{s}] acc={m['accuracy']:.3f} prec={m['precision']:.3f} recall={m['recall']:.3f} "
              f"spec={m['specificity']:.3f} f1={m['f1']:.3f} bal_acc={m['balanced_accuracy']:.3f}")
        print(f"       TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}   "
              f"(predict-all-abnormal baseline accuracy = {y.mean():.3f}; python/C++ agreement {agree:.4f})")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "classifier_metrics.json").write_text(json.dumps(report, indent=2))
    print("\nsaved results/classifier_metrics.json")

if __name__ == "__main__":
    main()
