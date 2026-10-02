"""One command per capture:

    python -m roomscan <capture_path> [--out DIR] [--tier lidar|video|photo] [--no-drift] [--no-damage]

<capture_path> may be a Stray Scanner export folder (LiDAR tier), a .mp4/.mov file or a folder
holding one (video tier), or a folder of per-room photo folders (photo tier). The tier is
auto-detected unless --tier is given (e.g. --tier video on a Stray folder uses only rgb.mp4).
"""
import argparse
import os

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="roomscan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tier", choices=["lidar", "video", "photo"], default=None)
    ap.add_argument("--id", default=None, help="capture id used in the output")
    ap.add_argument("--no-drift", action="store_true", help="ablation: use ARKit poses as-is")
    ap.add_argument("--no-damage", action="store_true")
    a = ap.parse_args()
    cid = a.id or os.path.basename(os.path.normpath(a.capture))
    out = a.out or os.path.join("out", f"{cid}{'_' + a.tier if a.tier else ''}")
    run(a.capture, out, tier=a.tier, capture_id=cid, opts={"drift": not a.no_drift}, damage=not a.no_damage)


if __name__ == "__main__":
    main()
