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

## Result (regenerated end to end from raw captures by `fixloop/reproduce.sh`)

| metric | before | predicted | **after** |
|---|---|---|---|
| walls passing (1 cm / 0.5 %) | 4 / 22 (18 %) | ≥ 11 / 22 | **5 / 20 (25 %)** |
| median absolute length difference | 6.7 cm | ≤ 2.0 cm | **6.2 cm** |
| p90 absolute length difference | 16.4 cm | — | **17.1 cm** |
| unsupported polygon edges (with_ceiling / floor_only) | 22 / 32 | ≤ 7 / 11 | **16 / 18** |
| gate | FAIL | FAIL | **FAIL** |

Readable diff: `git diff fixloop-before fixloop-after -- roomscan/` (60 changed lines, also written to `out/fixloop/fix.diff`). Overlays: `out/fixloop/<tag>/overlay.png`.

## Post-mortem: the prediction was badly wrong

**What the fix did.** It removed what it targeted. Unsupported edges dropped by 27 % and 44 % (not the predicted ⅔), and the polygons got simpler: 77/84 edges became 69/66. Rooms whose wall set was already stable became exactly repeatable. R2 went to four walls within 1 cm.

**Why the headline barely moved.** The hypothesis was right about the *mechanism*: wrong neighbour edges, not surface noise. It was wrong about *which* wrong edges dominate. We split every matched wall by how far its two neighbouring walls moved between the captures (`scripts/decompose_repeat.py`, same numbers before and after):

| class (neighbour shift between captures) | walls before / after | median abs length difference | passing |
|---|---|---|---|
| both neighbours are the same surface (< 5 cm) | 6 / 6 | 0.3 / 0.7 cm | 4 / 5 |
| a neighbour moved 5–15 cm | 11 / 10 | 7.1 / 8.9 cm | 0 / 0 |
| a neighbour moved ≥ 15 cm | 5 / 4 | 15.4 / 12.2 cm | 0 / 0 |

1. **The LiDAR wall measurement itself is repeatable at the gate level.** When both captures end a wall at the same two physical surfaces, lengths agree to well under 1 cm.
2. **Every failing wall has a neighbour that moved 5–30 cm.** These neighbours *are* supported by LiDAR points, so an evidence filter cannot remove them. Two causes:
   - **5–15 cm: the snap picks a different surface layer.** The "outermost dense layer" rule lands on the wall in one capture and on furniture or a cabinet face in the other, when one capture saw the wall above the furniture and the other did not.
   - **≥ 15 cm: room extents segmented differently.** A corner region (R7, R3, R8 in the overlay) belongs to the room in one capture and not in the other.
3. **The 0.30 m jog cap was chosen on this same data** from a 5-value sweep (0.10–∞; all gave 3–5 passing walls). The choice does not change the conclusion.

**What would move the gate next** (not shipped in this loop):
- Make layer selection consistent by preferring the layer with points above 1.8 m, where furniture rarely reaches, over the outermost layer.
- Snap room extents to structural walls rather than to floor evidence.

The decomposition predicts the size of each lever: it bounds the reachable pass rate by the share of walls in each class.

## Reproduce

```bash
bash fixloop/reproduce.sh     # checks out fixloop-before and fixloop-after into worktrees, runs both LiDAR captures, compares
```
