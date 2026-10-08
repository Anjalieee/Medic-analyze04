# Auto-generated results summary (verify against the graphs and rewrite in your own words)

## E1 - thread scaling
Dataset: 1000 images, 512px. Sequential time: 6.225 s (160.7 images/s), peak memory 6 MB.

Sequential stage share: load 21%, resize 7%, denoise 9%, enhance 6%, segment 19%, roi 33%, features 6%
Estimated serial fraction of the within-image mode (load + ROI labelling): 53.6% -> Amdahl upper bound on speedup = 1.9x

### omp_images
| threads | time (s) | speedup | efficiency | Karp-Flatt e |
|---|---|---|---|---|
| 1 | 6.258 | 0.99 | 0.99 | - |
| 2 | 3.085 | 2.02 | 1.01 | -0.009 |
| 4 | 2.019 | 3.08 | 0.77 | 0.099 |
| 8 | 1.595 | 3.90 | 0.49 | 0.150 |
| 10 | 1.550 | 4.02 | 0.40 | 0.166 |

Best speedup 4.02x at 10 threads (efficiency 40%); at 10 threads: 4.02x, efficiency 40%.
Efficiency first drops below 70% at 8 threads.
Marginal gain when adding threads: 1->2: +103%, 2->4: +53%, 4->8: +27%, 8->10: +3%

### omp_pixels
| threads | time (s) | speedup | efficiency | Karp-Flatt e |
|---|---|---|---|---|
| 1 | 7.660 | 0.81 | 0.81 | - |
| 2 | 5.651 | 1.10 | 0.55 | 0.816 |
| 4 | 4.549 | 1.37 | 0.34 | 0.641 |
| 8 | 4.539 | 1.37 | 0.17 | 0.691 |
| 10 | 4.565 | 1.36 | 0.14 | 0.704 |

Best speedup 1.37x at 8 threads (efficiency 17%); at 10 threads: 1.36x, efficiency 14%.
Efficiency first drops below 70% at 2 threads.
Marginal gain when adding threads: 1->2: +36%, 2->4: +24%, 4->8: +0%, 8->10: -1%

Overhead check: omp_pixels with 1 thread is 1.23x the sequential time (1.00 = no OpenMP overhead).

## E2 - dataset-size scaling
| images | seq (s) | omp_images (s) | S | omp_pixels (s) | S | seq img/s | omp_images img/s | peak MB (seq) | peak MB (omp_images) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0.63 | 0.11 | 5.90 | 0.45 | 1.39 | 158.1 | 955.5 | 5 | 26 |
| 500 | 3.21 | 0.62 | 5.14 | 2.33 | 1.38 | 155.9 | 801.4 | 6 | 29 |
| 1000 | 6.38 | 1.19 | 5.38 | 4.71 | 1.36 | 156.8 | 843.5 | 7 | 32 |
| 5000 | 33.28 | 5.66 | 5.88 | 22.87 | 1.46 | 150.3 | 884.1 | 8 | 32 |
| 10000 | 33.68 | 5.15 | 6.54 | 26.07 | 1.29 | 296.9 | 1942.7 | 10 | 32 |

seq: time ~ N^0.91 (1.00 = perfectly linear in number of images)

omp_images: time ~ N^0.88 (1.00 = perfectly linear in number of images)

omp_pixels: time ~ N^0.91 (1.00 = perfectly linear in number of images)

## E3 - image-size scaling
| side (px) | seq (s) | omp_images (s) | S | omp_pixels (s) | S | peak MB (seq) | peak MB (omp_images) |
|---|---|---|---|---|---|---|---|
| 256 | 0.33 | 0.04 | 7.87 | 0.30 | 1.09 | 3 | 10 |
| 512 | 1.25 | 0.15 | 8.35 | 0.91 | 1.38 | 5 | 28 |
| 1024 | 4.82 | 0.96 | 5.02 | 3.07 | 1.57 | 16 | 99 |
| 2048 | 9.86 | 1.59 | 6.20 | 6.22 | 1.59 | 42 | 323 |

seq: time ~ (side)^1.67  (2.00 = linear in pixel count)

omp_images: time ~ (side)^1.84  (2.00 = linear in pixel count)

omp_pixels: time ~ (side)^1.49  (2.00 = linear in pixel count)

