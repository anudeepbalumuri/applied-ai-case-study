import sys; sys.path.insert(0, '.')
import numpy as np, networkx as nx
from roomscan.capture import load_capture
from roomscan import mono as M
from roomscan.photo import pair_pose
cap = load_capture(sys.argv[1])
frames, fps = M.select_video_frames(cap.video_path, M.VIDEO_FPS, M.MAX_FRAMES)
net, lm = M.DepthNet.get(), M.LearnedMatcher.get()
D = [net(f["img"]) / M.MONO_SCALE_BIAS for f in frames]; F = [lm.features(f["img"]) for f in frames]
h, w = frames[0]["img"].shape[:2]
K = np.array([[M.IPHONE_FOCAL_FACTOR * max(w, h), 0, w / 2], [0, M.IPHONE_FOCAL_FACTOR * max(w, h), h / 2], [0, 0, 1.0]])
n = len(frames); G = nx.Graph(); G.add_nodes_from(range(n))
for i in range(n):
    for d in range(1, 4):
        if i + d < n and pair_pose(F[i], F[i + d], D[i], D[i + d], K, K, lm) is not None:
            G.add_edge(i, i + d)
comps = sorted(nx.connected_components(G), key=len, reverse=True)
print("components (size, range):", [(len(c), min(c), max(c)) for c in comps if len(c) > 2])
print("singletons:", sorted(k for c in comps if len(c) <= 2 for k in c))
# raw match counts across the biggest gaps
big = [c for c in comps if len(c) > 5]
for a in big:
    for b in big:
        if max(a) < min(b) and min(b) - max(a) < 20:
            i, j = max(a), min(b)
            print(f"gap {i}->{j}: matches {len(lm.match(F[i], F[j]))}, sharp {frames[i]['sharpness']:.0f}/{frames[j]['sharpness']:.0f}")
