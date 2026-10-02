# roomscan

roomscan turns an iPhone capture into a stitched, dimensioned floor plan. It handles three kinds of input (tiers): **LiDAR**, **video** and **photos**. Every capture produces the same output:
- per-room walls, ceiling height, floor area and openings, each with a 95 % interval
- room adjacency
- per-surface damage regions with their class and metric extent
- concealed-damage flags naming the rule that fired
- scope line items keyed to surfaces
- a rendered plan

Capture uses stock apps (Route 2). See [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md) and the [device matrix](docs/DEVICE_MATRIX.md).

## Fresh machine: about 10 minutes

```bash
git clone <repo> roomscan && cd roomscan
python3.11 -m venv .venv && . .venv/bin/activate      # or: conda create -p .venv python=3.11
pip install -r requirements.txt                       # ~4 GB with CUDA torch
python scripts/fetch_weights.py                       # ~3.5 GB of public weights -> ./weights
```

The pipeline was tested on Ubuntu (WSL2) with an RTX 5060 8 GB and CUDA 12.8 wheels. It also runs CPU-only (install the default torch wheel); the video, photo and damage stages are then about 5–10× slower. Nothing calls a hosted service.

## One command per capture

```bash
python -m roomscan <capture>              # tier auto-detected
python -m roomscan <capture> --tier video # force a tier (e.g. use only rgb.mp4 of a Stray folder)
```

| input | tier |
|---|---|
| Stray Scanner export folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) | LiDAR |
| `.mov` / `.mp4` file, or a folder holding one | video |
| folder of per-room photo folders (`kitchen/`, `hall/` …) | photo |

Output in `out/<capture>/`:
- **`plan.json`**: follows [schema/plan.schema.json](schema/plan.schema.json) and is validated on every run.
- **`plan.png` / `plan.svg`**: the rendered, dimensioned plan.

Options: `--out DIR`, `--no-drift` (ablation: ARKit poses used as-is), `--no-damage`.

## Reproduce every reported number

```bash
python -m bench.run_benchmark          # runs all capture x tier configs, writes out/bench/results.json + docs/BENCHMARK.md
python -m bench.make_photo_tier ...    # (called automatically) photo-tier proxy folders from LiDAR captures
bash fixloop/reproduce.sh              # fix loop: before and after runs from their git tags + diff
```

Raw data goes in `data/raw/` (gitignored, too large for git). Unzip the three Stray Scanner exports there; `bench/run_benchmark.py` lists the expected paths.

## Layout

```
roomscan/            pipeline package
  capture.py         input loading and tier detection
  fusion.py          LiDAR depth fusion (normals, flying-pixel rejection, chunked voxels)
  drift.py           submap pose graph + Manhattan / floor anchors (LiDAR drift correction)
  layout.py plan.py  room segmentation, rectilinear polygons, wall snapping, ceilings
  openings.py        doors / windows / passages with see-through and mirror tests
  mono.py            video tier (frame selection, densification, PnP pose graph, metric depth)
  posegraph.py       robust 6-DoF pose graph with Manhattan rotation anchors
  photo.py           photo tier (pairwise PnP graph, box layout, cross-room stitching)
  damage.py          damage detection -> surfaces, concealed-damage rules, scope
  uncertainty.py     error budget -> intervals (calibration.json holds per-tier multipliers)
  output.py render.py  JSON assembly and plan rendering
bench/               benchmark, comparison, ground-truth gates, report
docs/                protocol, device matrix, benchmark, technical report, compliance matrix, fix loop
schema/              published output schema
scripts/             diagnostics used during development (kept for the record)
```
