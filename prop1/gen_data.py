"""
Proposition 1 -- dataset v2 (analytic, no Blender).

Objects = compositions of tube/ring parts so that voxel skeletons are curves.
GT contact graph = 1-voxel dilation adjacency (>=2 overlap voxels);
constructor-declared edges are printed alongside as a cross-check.

Hull theorem this dataset is built around: a voxel outside the object survives
the visual hull iff it has a clear ray along at least one axis, so hull error
== enclosed pocket.  Wireframe objects are hull-easy; jar_lid and box_in_box
contain enclosed pockets between parts (hull bridges them; per-part views do not).

Output per object (.npz): label, occ, proj (3 joint views), part_projs (JSON),
gt_edges, declared_edges, hull_iou, kind, seed + manifest.json.
"""
import numpy as np
import json
import os
from scipy import ndimage

RES = 64
GRID_MIN, GRID_MAX = -1.0, 1.0
PITCH = (GRID_MAX - GRID_MIN) / RES
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------------- primitives (masks on grid) ----------------

def capsule(X, Y, Z, p0, p1, r):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    ll = d @ d
    t = ((X - p0[0]) * d[0] + (Y - p0[1]) * d[1] + (Z - p0[2]) * d[2]) / max(ll, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    dx = X - (p0[0] + t * d[0]); dy = Y - (p0[1] + t * d[1]); dz = Z - (p0[2] + t * d[2])
    return dx * dx + dy * dy + dz * dz <= r * r


def ring(X, Y, Z, c, axis, R, r):
    ax = [X, Y, Z]; u, v = [k for k in range(3) if k != axis]
    du = ax[u] - c[u]; dv = ax[v] - c[v]; dw = ax[axis] - c[axis]
    q = np.sqrt(du * du + dv * dv) - R
    return q * q + dw * dw <= r * r


def box(X, Y, Z, c, half):
    return ((np.abs(X - c[0]) <= half[0]) & (np.abs(Y - c[1]) <= half[1]) &
            (np.abs(Z - c[2]) <= half[2]))


def cyl(X, Y, Z, c, axis, R, h):
    ax = [X, Y, Z]; u, v = [k for k in range(3) if k != axis]
    du = ax[u] - c[u]; dv = ax[v] - c[v]; dw = ax[axis] - c[axis]
    return (du * du + dv * dv <= R * R) & (np.abs(dw) <= h)


# ---------------- contact graph ----------------

def contact_graph(parts, min_vox=2, dil=1):
    edges = []
    for i in range(len(parts)):
        di = ndimage.binary_dilation(parts[i]["mask"], iterations=dil)
        for j in range(i + 1, len(parts)):
            if np.count_nonzero(di & parts[j]["mask"]) >= min_vox:
                edges.append((i, j))
    return edges


def visual_hull_from(occ, res=RES):
    """Joint three-view hard intersection (condition C4 input)."""
    px = occ.any(axis=0)  # (y,z)
    py = occ.any(axis=1)  # (x,z)
    pz = occ.any(axis=2)  # (x,y)
    H = px[None, :, :] & py[:, None, :] & pz[:, :, None]
    return H


# ---------------- object constructors ----------------
# each: (parts, declared_edges); part = dict(name, lid, mask)

def obj_cage_cup(X, Y, Z, r):
    R = r.uniform(0.26, 0.32)
    zt = r.uniform(0.32, 0.36); zb = -zt
    rt = r.uniform(0.045, 0.055)
    posts = []
    n = 3
    for k in range(n):
        th = k * 2 * np.pi / n
        xy = (R * np.cos(th), R * np.sin(th))
        posts.append(capsule(X, Y, Z, (*xy, zb), (*xy, zt), rt))
    hR = r.uniform(0.14, 0.16)
    handle = ring(X, Y, Z, (R + 0.10, 0, 0), 2, hR, 0.05)  # xz-plane torus
    parts = [dict(name="ring_top", lid=1, mask=ring(X, Y, Z, (0, 0, zt), 2, R, rt)),
             dict(name="ring_bot", lid=2, mask=ring(X, Y, Z, (0, 0, zb), 2, R, rt))]
    for k in range(n):
        parts.append(dict(name=f"post{k}", lid=3 + k, mask=posts[k]))
    parts.append(dict(name="handle", lid=6, mask=handle))
    declared = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 5)]
    return parts, declared


def obj_chair(X, Y, Z, r):
    h = r.uniform(0.25, 0.30)          # half side of seat
    zs = 0.10
    rf = 0.045
    seat = (capsule(X, Y, Z, (-h, -h, zs), (h, -h, zs), rf) |
            capsule(X, Y, Z, (h, -h, zs), (h, h, zs), rf) |
            capsule(X, Y, Z, (h, h, zs), (-h, h, zs), rf) |
            capsule(X, Y, Z, (-h, h, zs), (-h, -h, zs), rf))
    leg_r = 0.05
    z_leg_top, z_leg_bot = zs, -0.50
    parts = [dict(name="seat", lid=1, mask=seat)]
    d = h - 0.06
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        parts.append(dict(name=f"leg{k}", lid=2 + k,
                          mask=capsule(X, Y, Z, (sx * d, sy * d, z_leg_bot),
                                       (sx * d, sy * d, z_leg_top), leg_r)))
    zb0, zb1 = zs, r.uniform(0.50, 0.58)
    back_r = 0.045
    parts.append(dict(name="backL", lid=6, mask=capsule(X, Y, Z, (-d, -d, zb0), (-d, -d, zb1), back_r)))
    parts.append(dict(name="backR", lid=7, mask=capsule(X, Y, Z, (d, -d, zb0), (d, -d, zb1), back_r)))
    parts.append(dict(name="topbar", lid=8, mask=capsule(X, Y, Z, (-d, -d, zb1), (d, -d, zb1), back_r)))
    # seat0, legs1-4, backL5, backR6, topbar7  (verified against voxel adjacency)
    declared = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                (4, 5), (2, 6), (5, 7), (6, 7)]
    return parts, declared


def obj_table(X, Y, Z, r):
    h = r.uniform(0.35, 0.42)
    zt = r.uniform(0.35, 0.45)
    rf = 0.045
    top = (capsule(X, Y, Z, (-h, -h, zt), (h, -h, zt), rf) |
           capsule(X, Y, Z, (h, -h, zt), (h, h, zt), rf) |
           capsule(X, Y, Z, (h, h, zt), (-h, h, zt), rf) |
           capsule(X, Y, Z, (-h, h, zt), (-h, -h, zt), rf))
    parts = [dict(name="frame", lid=1, mask=top)]
    d = h - 0.07
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        parts.append(dict(name=f"leg{k}", lid=2 + k,
                          mask=capsule(X, Y, Z, (sx * d, sy * d, -0.50),
                                       (sx * d, sy * d, zt), 0.05)))
    declared = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return parts, declared


def obj_bracket(X, Y, Z, r):
    v = capsule(X, Y, Z, (-0.35, 0, -0.40), (-0.35, 0, 0.40), 0.06)
    hh = capsule(X, Y, Z, (-0.35, 0, -0.40), (0.35, 0, -0.40), 0.06)
    b = capsule(X, Y, Z, (-0.35, 0, 0.05), (0.15, 0, -0.40), 0.05)
    parts = [dict(name="vert", lid=1, mask=v),
             dict(name="horiz", lid=2, mask=hh),
             dict(name="brace", lid=3, mask=b)]
    declared = [(0, 1), (0, 2), (1, 2)]
    return parts, declared


def obj_spring(X, Y, Z, r):
    R = r.uniform(0.24, 0.28)
    p = r.uniform(0.15, 0.18)          # pitch per turn
    turns = 3
    n = 60
    t = np.linspace(0, turns * 2 * np.pi, n)
    z0, z1 = -0.30, 0.30
    zs = np.linspace(z0, z1, n)
    pts = np.stack([R * np.cos(t), R * np.sin(t), zs], axis=1)
    m = np.zeros_like(X, dtype=bool)
    for i in range(n - 1):
        m |= capsule(X, Y, Z, pts[i], pts[i + 1], 0.05)
    parts = [dict(name="coil", lid=1, mask=m)]
    return parts, []


def obj_jar_lid(X, Y, Z, r):
    # jar: solid cylinder r=0.30, z in [-0.40, 0.30], carved cavity r=0.24 from z=-0.30 up
    cav = cyl(X, Y, Z, (0, 0, 0.35), 2, 0.24, 0.65)   # z in [-0.30, 1.0]
    jar = cyl(X, Y, Z, (0, 0, -0.05), 2, 0.30, 0.35) & ~cav
    lid = box(X, Y, Z, (0, 0, 0.215), (0.11, 0.11, 0.05))
    # square lid: diagonal corner radius 0.11*sqrt(2)=0.156 -> clearance to
    # inner wall (0.24) = 0.084 ≈ 2.7 voxels, keeps GT adjacency and 26-connectivity apart
    parts = [dict(name="jar", lid=1, mask=jar),
             dict(name="lid", lid=2, mask=lid)]
    declared = []
    return parts, declared


def obj_box_in_box(X, Y, Z, r):
    w = r.uniform(0.33, 0.37)
    inner_w = w - 0.12 - 0.09          # wall 0.12, clearance 0.09
    outer = box(X, Y, Z, (0, 0, 0), (w, w, w)) & ~box(X, Y, Z, (0, 0, 0), (w - 0.12, w - 0.12, w - 0.12))
    inner = box(X, Y, Z, (0, 0, 0), (inner_w, inner_w, inner_w))
    parts = [dict(name="shell", lid=1, mask=outer),
             dict(name="core", lid=2, mask=inner)]
    declared = []
    return parts, declared


CONSTRUCTORS = {
    "cage_cup": obj_cage_cup, "chair": obj_chair, "table": obj_table,
    "bracket": obj_bracket, "spring": obj_spring,
    "jar_lid": obj_jar_lid, "box_in_box": obj_box_in_box,
}


def build_object(kind, seed, res=RES):
    r = np.random.default_rng(seed)
    xs = np.linspace(GRID_MIN, GRID_MAX, res, endpoint=False) + PITCH / 2
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    parts, declared = CONSTRUCTORS[kind](X, Y, Z, r)
    label = np.zeros((res, res, res), dtype=np.int8)
    for p in parts:
        label[p["mask"]] = p["lid"]
    occ = label > 0
    gt_edges = contact_graph(parts)
    # joint projections
    proj = [occ.any(axis=0), occ.any(axis=1), occ.any(axis=2)]
    # per-part projections (oracle-colored views -> per-part silhouettes)
    part_projs = {}
    for p in parts:
        m = label == p["lid"]
        part_projs[p["name"]] = [m.any(axis=0), m.any(axis=1), m.any(axis=2)]
    H = visual_hull_from(occ)
    hull_iou = float((H & occ).sum() / max((H | occ).sum(), 1))
    return dict(kind=kind, seed=seed, label=label, occ=occ, proj=proj,
                parts=[dict(name=p["name"], lid=p["lid"]) for p in parts],
                gt_edges=gt_edges, declared=declared, part_projs=part_projs,
                hull_iou=hull_iou)


def main():
    os.makedirs(OUT, exist_ok=True)
    kinds = ["cage_cup", "chair", "table", "bracket", "spring", "jar_lid", "box_in_box"]
    n_per = int(os.environ.get("N_PER_KIND", "5"))
    manifest, seed = [], 5000
    for k in kinds:
        for i in range(n_per):
            seed += 1
            o = build_object(k, seed)
            mism = ""
            if sorted(map(tuple, o["gt_edges"])) != sorted(map(tuple, o["declared"])) and o["declared"]:
                mism = (f"  MISMATCH declared={sorted(map(tuple, o['declared']))}"
                        f" adj={sorted(map(tuple, o['gt_edges']))}")
            fn = os.path.join(OUT, f"{k}_{i:02d}.npz")
            np.savez_compressed(
                fn,
                label=o["label"], occ=o["occ"],
                proj=np.stack(o["proj"]).astype(np.uint8),
                gt_edges=np.array(o["gt_edges"], dtype=np.int64).reshape(-1, 2),
                declared_edges=np.array(o["declared"], dtype=np.int64).reshape(-1, 2),
                parts_json=np.array(json.dumps(o["parts"])),
                part_projs_json=np.array(json.dumps(
                    {n: [a.astype(np.uint8).tolist() for a in v] for n, v in o["part_projs"].items()})),
                hull_iou=np.array(o["hull_iou"]),
                kind=np.array(k), seed=np.array(seed),
            )
            print(f"{k}_{i:02d}: parts={len(o['parts'])} gt_edges={len(o['gt_edges'])} "
                  f"hull_iou={o['hull_iou']:.3f}{mism}")
            manifest.append(dict(file=os.path.basename(fn), kind=k, seed=seed,
                                 n_parts=len(o["parts"]), n_edges=len(o["gt_edges"]),
                                 hull_iou=o["hull_iou"]))
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest)} objects -> {OUT}")


if __name__ == "__main__":
    main()
