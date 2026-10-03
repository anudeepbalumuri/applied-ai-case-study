# Device matrix

| tier | runs on | capture tool | sensor data used | what the pipeline gets |
|---|---|---|---|---|
| **LiDAR** | iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max (any LiDAR iPhone works the same) | Stray Scanner (free) | 256×192 LiDAR depth + confidence, ARKit VIO poses, intrinsics, 1920×1440 RGB video | metric depth, metric poses (drift-corrected), gravity |
| **Video** | any iPhone 15 or newer (15, 15 Plus and up) | native Camera, 1080p30, HDR off | RGB video only | everything estimated: poses (PnP pose graph + Manhattan anchors), metric scale (mono-depth network + camera-height prior), gravity (Manhattan + trajectory) |
| **Photo** | any iPhone 15 or newer | native Camera, 1x, landscape | 2–8 stills per room (HEIC/JPEG), EXIF focal length | per-room relative poses (pairwise PnP on mono depth), metric scale (network + camera-height prior), room boxes, cross-room placement (image links, then door matching) |

## Accuracy each tier honestly delivers

Measured on our benchmark (`docs/BENCHMARK.md`). "vs LiDAR" means agreement with the LiDAR result on the same rooms. Tape ground truth is still pending, so these are agreement figures, not absolute accuracy.

| quantity | LiDAR | video | photo |
|---|---|---|---|
| wall length, repeatability between captures | see BENCHMARK, repeatability | — | — |
| wall length vs LiDAR (median / p90 relative) | reference | see BENCHMARK | see BENCHMARK |
| ceiling height | plane fit to observed ceiling; `null` + lower bound when the ceiling was never seen | weak (mono depth), wide interval | weak, wide interval |
| openings | detected with see-through and mirror tests | coarse | doors only where photographed |
| whole-property stitch | single world frame, drift-corrected | single world frame, drifts more | image links / door matching; unplaced rooms reported |
| typical 95 % interval on a 3 m wall | ±1–2 cm | about ±10–20 cm | about ±25–40 cm |

The intervals widen with the scale uncertainty of each tier (`roomscan/uncertainty.py`). Per-tier multipliers in `roomscan/calibration.json` come from the coverage analysis in the technical report.
