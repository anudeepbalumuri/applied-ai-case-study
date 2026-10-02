"""Pose graph for camera-only tiers (video, photo).

Edges come from PnP on metric mono depth (photo.pair_pose): consecutive frames (odometry) and gated
long-range pairs (loops). Rotation drift is removed with Manhattan anchors: every frame's own depth
normals measure its rotation relative to the room's three dominant axes (plane-anchored correction),
which is exactly the information VIO-free video lacks.

Solve: 6-DoF per node (rotation vector + translation), robust (soft-L1) sparse least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def chain_init(n, edges):
    """Initial poses by BFS over the max-inlier spanning forest. edges: {(i,j): {'T': T_i_j, 'w': inliers}}."""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), e in edges.items():
        G.add_edge(i, j, w=e["w"])
    poses = {}
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    for comp in comps:
        root = min(comp)
        poses[root] = np.eye(4)
        T = nx.maximum_spanning_tree(G.subgraph(comp), weight="w")
        for a, c in nx.bfs_edges(T, root):
            Tac = edges[(a, c)]["T"] if (a, c) in edges else np.linalg.inv(edges[(c, a)]["T"])
            poses[c] = poses[a] @ Tac
    return poses, comps


def frame_axis_obs(normals_cam, pred_axes, max_deg=10.0, min_frac=0.12):
    """Per-axis Manhattan observations for one frame.

    pred_axes: world Manhattan axes expressed in this camera's frame (3x3 columns) from the current
    estimate. Each well-supported normal cluster within `max_deg` of a predicted axis (either sign)
    becomes an observation (measured unit vector in camera frame, world axis index, sign).
    """
    obs = []
    if len(normals_cam) < 200:
        return obs
    c = np.cos(np.deg2rad(max_deg))
    for k in range(3):
        a = pred_axes[:, k]
        d = normals_cam @ a
        for sgn in (1.0, -1.0):
            sel = normals_cam[d * sgn > c]
            if len(sel) < min_frac * len(normals_cam):
                continue
            m = sel.mean(0)
            m /= np.linalg.norm(m)
            obs.append((m, k, sgn, len(sel) / len(normals_cam)))
    return obs


def frame_axes(normals_cam, min_pts=400):
    """Dominant orthogonal normal directions in one frame (camera coords), or None."""
    from .mono import manhattan_axes
    if len(normals_cam) < min_pts:
        return None
    return manhattan_axes(normals_cam, iters=4)


def _match_axes(Rm, Rp, max_deg=12.0):
    """Best signed-permutation alignment of measured axes Rm (3x3 cols) to predicted Rp. Returns R_corr or None.

    Only the axes that are well constrained matter; we return the rotation taking predicted to measured.
    """
    best, best_err = None, np.inf
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = Rm[:, perm] * np.array(signs)
            if np.linalg.det(M) * np.linalg.det(Rp) < 0:
                continue
            Rc = M @ Rp.T
            ang = np.degrees(np.linalg.norm(Rotation.from_matrix(Rc).as_rotvec()))
            if ang < best_err:
                best, best_err = Rc, ang
    return (best, best_err) if best_err < max_deg else (None, best_err)


def optimise(n, edges, init, anchors=None, W=None, sigma_r=np.deg2rad(1.0), sigma_t=0.03,
             sigma_anchor=np.deg2rad(1.5), iters=60):
    """edges: {(i,j): {'T': T_i_j (metric), 'w': inliers, 'loop': bool}}; anchors: {i: R_cam_axes (3x3)}.

    W: world Manhattan axes (3x3, columns); an anchor says R_wc^T @ W ~= measured axes (up to the
    signed permutation resolved beforehand -> anchors hold the resolved camera-frame axes).
    """
    nodes = sorted(init)
    idx = {k: a for a, k in enumerate(nodes)}
    x0 = np.zeros(6 * len(nodes))
    for k in nodes:
        a = idx[k]
        x0[6 * a:6 * a + 3] = Rotation.from_matrix(init[k][:3, :3]).as_rotvec()
        x0[6 * a + 3:6 * a + 6] = init[k][:3, 3]
    E = [(idx[i], idx[j], e) for (i, j), e in edges.items() if i in idx and j in idx]
    # anchors: {frame: [(measured unit normal in camera frame, world axis index, sign, support), ...]}
    A = [(idx[k], m, kk, sg, sup) for k, ol in (anchors or {}).items() if k in idx for (m, kk, sg, sup) in ol]
    root = 0

    def unpack(x, a):
        return Rotation.from_rotvec(x[6 * a:6 * a + 3]), x[6 * a + 3:6 * a + 6]

    def resid(x):
        r = []
        for a, b, e in E:
            Ra, ta = unpack(x, a)
            Rb, tb = unpack(x, b)
            Rm = Rotation.from_matrix(e["T"][:3, :3])
            tm = e["T"][:3, 3]
            Rrel = Ra.inv() * Rb
            trel = Ra.inv().apply(tb - ta)
            wgt = np.sqrt(min(e["w"], 400) / 100.0)
            r.extend((Rm.inv() * Rrel).as_rotvec() / sigma_r * wgt)
            r.extend((trel - tm) / sigma_t * wgt)
        for a, m, kk, sg, sup in A:
            Ra, _ = unpack(x, a)
            pred = Ra.inv().apply(W[:, kk] * sg)         # world axis seen in the camera frame
            r.extend(np.cross(pred, m) / sigma_anchor * np.sqrt(min(sup, 0.5) / 0.25))
        Rr, tr = unpack(x, root)
        r.extend((Rr.as_rotvec() - x0[0:3]) * 1e3)
        r.extend((tr - x0[3:6]) * 1e3)
        return np.array(r)

    m = 6 * len(E) + 3 * len(A) + 6
    J = lil_matrix((m, len(x0)), dtype=int)
    row = 0
    for a, b, _ in E:
        J[row:row + 6, 6 * a:6 * a + 6] = 1
        J[row:row + 6, 6 * b:6 * b + 6] = 1
        row += 6
    for a, *_ in A:
        J[row:row + 3, 6 * a:6 * a + 3] = 1
        row += 3
    J[row:row + 6, 0:6] = 1
    sol = least_squares(resid, x0, jac_sparsity=J, loss="soft_l1", f_scale=2.0, max_nfev=iters, x_scale="jac")
    out = {}
    for k in nodes:
        Ra, ta = unpack(sol.x, idx[k])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Ra.as_matrix(), ta
        out[k] = T
    # per-edge residual summary (for loop rejection / reporting)
    return out, {"cost": float(sol.cost), "nfev": int(sol.nfev), "edges": len(E), "anchors": len(A)}
