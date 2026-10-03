#!/usr/bin/env bash
# Fix loop reproduction: regenerate BEFORE and AFTER from raw captures and score both with the same code.
#   bash fixloop/reproduce.sh
# Needs: data/raw/{with_ceiling,floor_only} (see README), .venv, weights/ (not needed: runs with --no-damage).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/out/fixloop"
mkdir -p "$OUT"
for TAG in fixloop-before fixloop-after; do
  WT="$ROOT/.fixloop_wt/$TAG"
  if [ ! -d "$WT" ]; then
    git -C "$ROOT" worktree add -f --detach "$WT" "$TAG" >/dev/null
  fi
  mkdir -p "$WT/data"
  ln -sfn "$ROOT/data/raw" "$WT/data/raw"
  for CAP in with_ceiling floor_only; do
    case $CAP in
      with_ceiling) SRC=data/raw/with_ceiling/c7d28f72c6 ;;
      floor_only)   SRC=data/raw/floor_only/1a8384c3f6 ;;
    esac
    echo ">> $TAG $CAP"
    (cd "$WT" && "$ROOT/.venv/bin/python" -m roomscan "$SRC" --tier lidar --id "$CAP" --no-damage --out "$OUT/$TAG/$CAP" | tail -1)
  done
done
cd "$ROOT"
for TAG in fixloop-before fixloop-after; do
  echo "== $TAG"
  .venv/bin/python -m bench.compare "$OUT/$TAG/with_ceiling/plan.json" "$OUT/$TAG/floor_only/plan.json" > "$OUT/$TAG/compare.txt"
  .venv/bin/python - "$OUT/$TAG" <<'PY'
import json, sys
from bench.compare import compare
d = sys.argv[1]
c = compare(f"{d}/with_ceiling/plan.json", f"{d}/floor_only/plan.json")
s = c["summary"]
E = lambda p: [w for r in json.load(open(p))["rooms"] for w in r["walls"]]
ea, eb = E(f"{d}/with_ceiling/plan.json"), E(f"{d}/floor_only/plan.json")
res = {"walls_matched": s["walls_matched"], "walls_pass": s["walls_pass"],
       "median_abs_delta_m": s["median_abs_delta_m"], "p90_abs_delta_m": s["p90_abs_delta_m"],
       "unsupported_edges": [sum(not w["measured_from_points"] for w in ea), sum(not w["measured_from_points"] for w in eb)],
       "edges": [len(ea), len(eb)]}
json.dump(res, open(f"{d}/summary.json", "w"), indent=1)
print(json.dumps(res))
PY
  .venv/bin/python -m bench.overlay "$OUT/$TAG/with_ceiling/plan.json" "$OUT/$TAG/floor_only/plan.json" "$OUT/$TAG/overlay.png"
done
git -C "$ROOT" diff fixloop-before fixloop-after -- roomscan/ > "$OUT/fix.diff"
echo "diff: $OUT/fix.diff  ($(grep -c '^[+-][^+-]' "$OUT/fix.diff") changed lines)"
