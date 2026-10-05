# Compliance matrix

Status key:
- **Done**: implemented and exercised on our data.
- **Partial**: implemented, with limits stated.
- **Fail**: measured and failing; the number is in BENCHMARK.
- **Needs capture**: needs physical work (captures, tape) that a pipeline cannot supply.

| # | Requirement (brief) | File / path | Artifact | Status |
|---|---|---|---|---|
| **Part 1** | | | | |
| 1.1 | Capture route: stock app plus a one-page protocol | `docs/CAPTURE_PROTOCOL.md` | protocol for Stray Scanner, Camera video and Camera stills | Done |
| 1.2 | Photo tier: 2–8 stills/room, no depth or poses, produces a stitched plan | `roomscan/photo.py` | pairwise PnP graph, box layout, graph stitching | Partial: runs end to end; stitch fails on the proxy data (BENCHMARK) |
| 1.3 | Video tier: handheld clip | `roomscan/mono.py`, `roomscan/posegraph.py` | PnP pose graph, Manhattan anchors, densification | Partial: runs; **fails**: footprint 49–80 % short, 3.3 h on the long clip |
| 1.4 | LiDAR tier: depth, poses, intrinsics | `roomscan/capture.py`, `fusion.py`, `drift.py` | Stray Scanner loader, fusion, drift correction | Done |
| 1.5 | Device matrix, with honest per-tier accuracy | `docs/DEVICE_MATRIX.md` | table | Done |
| **Part 2: output contract** | | | | |
| 2.1 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `roomscan/plan.py`, `layout.py`, `openings.py`, `output.py` | `plan.json` rooms[] | Done (ceiling is `null` + lower bound when never observed) |
| 2.2 | Stitched multi-room plan with adjacency | `plan.py::_adjacency`, `photo.py` stitching | `property.adjacency` | Done (LiDAR/video); Partial (photo) |
| 2.3 | Per-surface damage regions with class and metric extent | `roomscan/damage.py` | `rooms[].damage[]` | Partial: works and returns 0 phantoms on undamaged captures; recall only tested with synthetic overlays |
| 2.4 | Concealed-damage flags with the rule that fired | `damage.py::RULES, apply_rules` | `rooms[].concealed_damage_flags[]`, `rules[]` | Done (rules); never fired on our data, which has no damage |
| 2.5 | Scope line items keyed to surfaces | `damage.py::build_scope` | `scope_line_items[]` | Done |
| 2.6 | Confidence interval on every measurement | `roomscan/uncertainty.py` | `value / sigma / ci95` everywhere | Done |
| 2.7 | One command per capture | `roomscan/__main__.py` | `python -m roomscan <capture>` | Done |
| 2.8 | JSON to the published schema | `schema/plan.schema.json`, `pipeline.py::_validate` | `schema_valid: true` | Done |
| 2.9 | Rendered plan | `roomscan/render.py` | `plan.png`, `plan.svg` | Done |
| **Benchmark set** | | | | |
| B.1 | Multi-room capture (3+ rooms plus a connector) | `data/raw/with_ceiling`, `floor_only` | 8–9 rooms incl. hallway | Done |
| B.2 | Furnished room with staged damage, two classes | — | — | **Needs capture** |
| B.3 | Same rooms at all three tiers, photo as per-room folders | `bench/make_photo_tier.py` | video = the captures' `rgb.mp4`; photo = **proxy** from video frames | Partial: real iPhone video and stills **need capture** |
| B.4 | A room captured twice at the same tier | `with_ceiling` vs `floor_only` (LiDAR) | repeatability table | Done |
| B.5 | Laser or tape ground truth on everything | `bench/gates.py` (template + scoring) | `ground_truth/*.json` | **Needs capture** (template generator ready) |
| **Gates** | | | | |
| G.1 | Opening widths ≤ 2 cm on ≥ 85 % (misses and phantoms count) | `bench/gates.py` | — | Pending ground truth |
| G.2 | Ceiling ≤ 1.5 cm; spread ≤ 1 cm; say biased vs unrepeatable | `bench/gates.py`, BENCHMARK ceiling table | — | Pending ground truth; spread not measurable (one capture never sees the ceiling) |
| G.3 | Repeatability within 1 cm / 0.5 % per wall | BENCHMARK repeatability | table | **Fail** (fix-loop target) |
| G.4 | Drift accountability plus on/off ablation | `roomscan/drift.py`, BENCHMARK drift | ablation table | Done |
| G.5 | Photo-tier whole-property stitch | BENCHMARK cross-tier | — | **Fail** on proxy data |
| G.6 | Photo ±8 %, video ±3 %, calibration at every tier | BENCHMARK cross-tier (coverage column) | — | **Fail**: no walls matched, so calibration is not possible (k = 1, uncalibrated) |
| **Part 3** | Head-to-head vs a consumer app, 2 rooms | `bench/head_to_head.py` | — | **Needs capture** (app export plus tape) |
| **Part 4** | Fix loop: declaration, shipped fix, before/after regenerable, diff | `docs/FIX_LOOP.md`, `fixloop/reproduce.sh`, tags `fixloop-before` / `fixloop-after` | `out/fixloop/` | Done (result in FIX_LOOP) |
| **Part 5** | Process evidence | git history | commits | Done (see note in the technical report) |
| **Deliverables** | | | | |
| D.3 | README to running in < 15 min | `README.md`, `requirements.txt`, `scripts/fetch_weights.py` | | Done (not timed on a clean machine) |
| D.4 | Reproduction bundle | `bench/run_benchmark.py`, `fixloop/reproduce.sh` | | Done |
| D.5 | Benchmark report | `docs/BENCHMARK.md` | | Done |
| D.7 | Technical report, ≤ 6 pages | `docs/TECHNICAL_REPORT.md` | | Done |
| D.8 | Raw benchmark data | `data/raw/` (3 Stray Scanner exports) | | Partial: no ground truth or app exports yet |
| **Constraints** | | | | |
| C.1 | Pretrained models disclosed; nothing calls our infrastructure | `scripts/fetch_weights.py` (list) | | Done |
| C.2 | Weights fetched by script | `scripts/fetch_weights.py` | | Done |
| C.3 | Mirrors, glass, wet-look surfaces, low light covered | `openings.py` (mirror and see-through tests), `damage.py` (negatives), `mono.py` (low-light count) | | Partial: handled in code; no dedicated test captures |
