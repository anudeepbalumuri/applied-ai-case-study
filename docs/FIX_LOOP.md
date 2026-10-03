# Fix loop

## Declaration (written 2026-10-03, before any fix code; baseline = tag `fixloop-before`)

### 1. Worst-performing gate and its failing number

**LiDAR repeatability.** Two LiDAR captures of the same multi-room floor (`with_ceiling`, `floor_only`) should give every matched wall within 1 cm or 0.5 %.

| baseline (`fixloop-before`) | value |
|---|---|
| walls passing | **4 / 22 (18 %)** |
| median absolute length difference | **6.7 cm** |
| p90 absolute length difference | 16.4 cm |
| rooms matched between captures | 5 (of 9 and 8) |

**Why this gate and not the photo-tier stitch.** The photo-tier stitch fails worse: 1 of 8 rooms placed, footprint −72 %. But that benchmark is a *proxy* built from video frames of the LiDAR walkthrough, and the proxy violates our own photo protocol: there are no overlapping doorway photos, and the frames are blurred. Its failure therefore mostly measures the proxy, and no change to the pipeline can be verified on it until real stills are captured. Among the gates measured on valid data, repeatability is the worst.

### 2. Root-cause hypothesis and evidence

**Hypothesis.** The wall-length differences come mainly from **polygon topology instability**, not from wall-surface position noise. Room polygons come from contours of the free-space mask. That mask picks up notches and jogs that are not walls:
- doorway cells grown into the room
- floor patches shadowed by furniture
- unobserved strips

Each jog splits a real wall into pieces and moves the endpoints of the walls next to it. The LiDAR snapping then measures the surfaces well, but it measures the *wrong set of edges*. The two captures produce different artifacts, so the same physical wall gets different lengths.

**Evidence** (from `scripts/repeat_fast.py` and the analysis in the session log):
- **Lengths disagree twice as much as surfaces do.** The surface position of matched walls differs by 3.3 cm median, but length by 6.7 cm.
- **Failing walls sit next to artifacts.** Of the 18 failing walls, 10 are adjacent to a jog shorter than 0.6 m and 7 are adjacent to an edge with no LiDAR support (`measured=False`).
- **The polygons are full of artifact edges.**
  - `with_ceiling`: 20 of 77 edges are shorter than 0.6 m and 22 have no LiDAR support.
  - `floor_only`: 31 of 84 edges are shorter than 0.6 m and 32 have no LiDAR support.
  - An edge with no wall points behind it is by construction not a measured wall.
- **The overlay shows it** (`bench.overlay`): both captures draw the same wall lines, but with different notches. Examples are the R3 top edge, the bottom of R2 (+24 cm) and the R1 left wall (−24.5 cm).

### 3. The fix and the predicted number

**Fix: evidence-based polygon regularisation** in `layout.py` / `plan.py`. An edge survives only if LiDAR points support it.
1. After snapping, an edge with no point support (`support < 40`) that lies between two parallel edges facing the same way is a jog or notch artifact. Collapse it: merge the two neighbours at the coordinate of the better-supported one (support-weighted, never averaged into an unsupported side).
2. Repeat until stable. Then recompute the extents and re-snap the merged walls once over their new full length.
3. Real steps such as niches, columns or chimney breasts keep their jog edge, because that edge's own face is measured.

**Predicted after the fix**, on the same two captures, rerun end to end:

| metric | before | predicted after |
|---|---|---|
| walls passing | 4 / 22 | **≥ 11 / 22 (≥ 50 %)** |
| median absolute length difference | 6.7 cm | **≤ 2.0 cm** |
| artifact edges (no support) | 22 / 32 | ≤ 1/3 of baseline |

**Honest expectation: the gate will probably still FAIL.** It requires every wall to pass. Walls where furniture hides the surface in one capture keep a few-cm surface difference that this fix does not touch. If the prediction holds, it is meaningful movement short of the gate, for the reason just stated.

## Result

*(filled after the fix ships: before/after table, readable diff, post-mortem)*

## Reproduce

```bash
bash fixloop/reproduce.sh     # checks out fixloop-before and fixloop-after into worktrees, runs both LiDAR captures, compares
```
