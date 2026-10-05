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
| wall length, repeatability (two captures) | 5/20 walls within 1 cm; median 6.2 cm. Walls with stable neighbours agree to 0.3–0.7 cm | — | — |
| footprint vs LiDAR | reference | 49–80 % short (partial coverage) | 69–76 % short (one room placed, proxy data) |
| ceiling height | plane fit when observed; `null` + lower bound otherwise | weak | weak |
| openings | detected with see-through and mirror tests | coarse | doors only where photographed |
| whole-property stitch | yes (drift-corrected) | partial: the chain breaks at blurred pans | fails on proxy data |
| honest summary | usable; repeatability gate not met | **not usable for measurement yet** | **not validated**: needs real protocol stills |

The intervals widen with the scale uncertainty of each tier (`roomscan/uncertainty.py`). Per-tier multipliers in `roomscan/calibration.json` come from the coverage analysis in the technical report.
