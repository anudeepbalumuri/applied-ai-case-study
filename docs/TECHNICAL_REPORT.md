# roomscan: technical report

*Max 6 pages. Numbers come from `docs/BENCHMARK.md`, which `python -m bench.run_benchmark` regenerates.*

## 1. Architecture

```
capture ──► tier reconstruction ──► gravity-aligned fused cloud ──► plan extraction ──► damage ──► JSON + plan
             LiDAR : ARKit poses → drift correction → depth fusion           │
             video : frames → DISK/LightGlue → PnP pose graph → mono depth   ├─ rooms, walls, ceilings, openings
             photo : per-room PnP graph → box layout → cross-room stitch     └─ adjacency
```

**One output contract for every tier.** Every tier ends in the same `plan.json` (validated against `schema/plan.schema.json`) and the same renderer. Tiers differ only in how the gravity-aligned metric point cloud and camera poses are produced. LiDAR and video share the plan extractor (`plan.py`). The photo tier uses a box-layout model per room, because 2–8 stills rarely observe enough floor to segment rooms.

**Plan extraction (LiDAR and video).**
1. Floor height comes from up-facing normals, and the Manhattan yaw from wall normals (4θ circular mean).
2. The cloud is rasterised at 2.5 cm into floor, wall and high-wall (>2.1 m) evidence.
3. Rooms are free space split by *structural* walls. Structural means long straight runs or wall points above 2.1 m; sofa backs and desks don't qualify. Door headers in the high band close doorways. Door-sized gaps along a wall line are bridged so that captures that never look up still separate rooms. Header-less constrictions are split by a distance-transform watershed.
4. Each room mask becomes a rectilinear polygon. Every edge is then **snapped to the wall surface measured by LiDAR points**: the outermost dense layer of a 5 mm histogram of inward-facing wall points, which is robust to furniture standing in front of the wall. That gives a sub-centimetre surface estimate with σ = MAD/√n_eff.
5. Ceilings: the highest strong horizontal layer of down-facing points inside the room. If the ceiling was never observed, the output is `null` with a lower bound. We don't guess.

**Openings.** For each wall we build an elevation occupancy image. Candidate gaps must pass two evidence tests:
- **See-through:** there must be points beyond the wall plane inside the gap. An unobserved wall patch has nothing behind it, so it can't become a phantom opening.
- **Mirror:** points behind a mirror, reflected back about the wall plane, land on in-room geometry. If they do, the gap is rejected as a mirror.

Widths are measured jamb to jamb from point percentiles.

**Damage.**
1. Grounding-DINO proposes regions for damage phrases.
2. CLIP verifies each one: the best positive phrase must beat every look-alike negative, including grout, cabinet seams, shadows, reflections and wet-look tiles.
3. SAM produces the mask.
4. Each mask pixel is **ray-cast onto the room's own plan surfaces**. That gives the surface id and metric extent with no depth sensor, so it works on all tiers; on LiDAR, sensor depth additionally rejects pixels occluded by furniture.
5. Views are made upright from the pose first, because portrait capture stores sideways frames and that wrecked detection.
6. Regions are fused across views. LiDAR and video need ≥ 2 views; floor "cracks" need much stronger evidence because of tile grout.
7. Seven explicit rules raise concealed-damage flags, each with the id of the rule that fired. Scope line items are keyed to surface ids, with intervals on quantities.

## 2. Tier design and device matrix

See `docs/DEVICE_MATRIX.md`.

- **LiDAR** trusts ARKit locally and corrects drift globally (§3).
- **Video** has no poses. We tried COLMAP incremental SfM first:
  - With SIFT it registered 48 of 215 frames.
  - With DISK + LightGlue it registered 192 of 215, but long-range matches on repetitive tiles glued rigid segments together at 65–170° errors.
  - Gating the loops made it correct (3° median) but fragmented (74 of 215).
  - With gap densification, mapping took over 75 CPU-minutes for a 37 s clip.

  The shipped design instead builds a **6-DoF pose graph from PnP on metric mono depth** (Depth-Anything-V2 Metric-Indoor, corrected by its bias measured against LiDAR):
  - Odometry edges to the next 1–3 frames, plus gated loop edges.
  - Raw-frame **densification across blurred whip-pans**, the only places the chain broke.
  - **Manhattan rotation anchors**: each frame's own depth normals measure its rotation relative to the room axes.
  - Per-frame depth scales fixed by pair ratios plus a prior tying each frame to the network's metric scale. Without the prior, the scale drifted +43 % over 480 edges.
  - Metric scale fuses the network with a handheld camera-height prior (1.42 ± 0.09 m) by inverse variance, inflated when the two cues disagree.
- **Photo**: a pairwise PnP graph per room, a box layout (farthest strong Manhattan wall in each direction, with unmeasured walls flagged σ = 35 cm), and graph stitching. Strong cross-room image links form clusters; door-width matching merges clusters with a no-overlap constraint. Rooms that can't be placed are listed, never silently overlapped.
- **Gravity** without an IMU: the vertical axis is the Manhattan axis that the trajectory varies least along and the optical axes are most perpendicular to. Its sign comes from up-facing surfaces outnumbering down-facing ones; a geometric below/above test can't decide the sign, because flipping the world swaps floor and ceiling and stays consistent. This replaced "camera y is down", which was wrong by 93° on portrait captures.

## 3. Drift handling (LiDAR)

ARKit poses are split into submaps of 40 keyframes. The 4-DoF pose graph (x, z, yaw per submap, with gravity trusted) has three kinds of term:
- **Odometry edges**, which keep consecutive submaps rigid.
- **Loop closures**: 2D point-to-line ICP between non-consecutive submaps that see the same walls, accepted only if inliers > 35 %, RMSE < 3.5 cm, |Δθ| < 5° and |t| < 40 cm.
- A **Manhattan anchor** on every submap's dominant wall direction.

The solver is robust (Huber) least squares. Vertical drift (5–9 cm measured) is removed by anchoring each submap's floor plane to the global floor height, smoothed and clamped to ±15 cm. The ablation (`--no-drift`) is in BENCHMARK. "Poses used as-is" is only the ablation baseline.

## 4. Error budget

Every interval is σ² = random + systematic + scale, then multiplied by a per-tier calibration factor:
- **Random:** point scatter / √n_eff, from the fit itself.
- **Systematic:** per-surface sensor bias; LiDAR 4 mm, video 15 mm, photo 30 mm.
- **Scale:** relative scale σ × length; LiDAR 0.2 %, video from the fused scale cues (3–8 %), photo 4.5 % or more.

Area σ propagates edge σ times edge length, plus 2 × scale. Unmeasured quantities get no point estimate (ceilings) or a widened σ (unseen photo walls).

## 5. Calibration analysis

`python -m bench.calibrate` fits a per-tier interval multiplier k = q95(|tier − LiDAR| / σ) / 1.96 over walls matched to the LiDAR pseudo-reference. **On our data there is nothing to calibrate on.** The video and photo tiers recover too few rooms for any wall to be matched against LiDAR (0 matched walls; `roomscan/calibration.json` stays at k = 1). The intervals for those tiers are therefore *uncalibrated priors* (§4), and they will be too narrow: footprint errors of 49–80 % are far outside any stated interval. This is the "confident garbage on thin input" failure the brief warns about, and we report it as such rather than hide it.

LiDAR intervals are not calibrated against tape either: ground truth is pending. Repeatability gives a lower bound on their error. Walls whose two neighbours are the same physical surfaces agree to 0.3–0.7 cm (`docs/FIX_LOOP.md`), which is consistent with the ±1–2 cm stated on a 3 m wall.

## 6. Fix loop

`docs/FIX_LOOP.md` has the full account.
- **Declared:** LiDAR repeatability, 4/22 walls passing, median 6.7 cm.
- **Hypothesis:** polygon-topology artifacts.
- **Shipped:** evidence-based polygon regularisation.
- **Result:** 5/20 walls, median 6.2 cm. The prediction (≥ 11/22, ≤ 2 cm) was **badly wrong**.
- **Post-mortem:** the wall-surface measurement is repeatable (0.3–0.7 cm). Failures come from *supported* but different neighbour surfaces: furniture-versus-wall layer choice, and room extents.
- **Reproduce:** `bash fixloop/reproduce.sh` regenerates both runs from raw captures.

## 6b. Results summary (BENCHMARK)

| gate | result |
|---|---|
| LiDAR repeatability (all walls ≤ 1 cm / 0.5 %) | **FAIL**: 5/20 pass, median 6.2 cm, p90 17.1 cm |
| Drift accountability | ablation reported. With drift correction on: 20 matched walls, 5 passing. Off: 11 matched, 4 passing. Footprint changes 2.6–3.2 m² per capture. |
| Video ±3 % | **FAIL**: 2–3 of 3–9 rooms recovered, footprint error 49–80 %, no walls match |
| Photo-tier stitch ±8 % | **FAIL**: 1 room placed per capture (proxy data), footprint error 69–76 % |
| Opening, ceiling and tape gates; head-to-head | **pending**: needs tape ground truth, an app export, and real video/photo captures |
| Damage phantoms on undamaged captures | 0 regions on all 3 LiDAR captures |
| Timing | LiDAR 1.4–3.6 min per capture; video 5–7 min on short clips but **3.3 h** on the 3.5-minute walkthrough (raw-frame densification over 87 breaks); photo < 1 min |

## 7. Known failure modes

- Mirrors are rejected as openings by the reflection test, but a mirror on a wall we never saw from the front can still leave a hole in the wall evidence.
- Glass shows up as LiDAR dropouts plus see-through points, so it's detected as a window.
- Wet-look floors are a CLIP negative for damage.
- Low light: frames with mean brightness < 45 are counted and reported. LiDAR still works; video and photo degrade.
- Open-plan spaces with no headers are split by the neck rule only if the constriction is clearly narrower than both sides.
- The photo tier needs overlapping photos and photographed doorways (protocol); otherwise rooms stay unplaced.
- Damage recall has only been checked with synthetic overlays. Our captures contain no real damage, and staged-damage captures are still required.
- **Video tier on long, fast walkthroughs:** blur breaks the chain at every pan. Densifying with raw frames reconnects it, but it costs hours on a 3.5-minute clip, and the pose graph still drifts. The output covers only part of the property.
- **The photo tier needs real stills that follow the protocol.** Our proxy (blurry video frames) gives partial room boxes and no cross-room links.

## 8. Process note

The repository was built in a few long AI-assisted sessions (Claude Code). The commit history is real and incremental, but compressed in time. Dead ends stay visible in the history: COLMAP SfM, the PnP chain without a scale prior, the camera-y gravity heuristic, the summed-positives damage verifier. The diagnostic scripts that found them are kept in `scripts/`.
