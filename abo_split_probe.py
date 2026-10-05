"""Probe ABO models for automatic part splittability (no manual Blender work).

Levels tried per model:
  L1 scene nodes / glTF primitives (= material groups; Blender counts these as materials)
  L2 connected components within each primitive (disjoint shells)
Report face-share tables so we can pick models whose components look like
body + lid/handle (a part with 3-40% of faces, not 0.1% debris).
"""
import os
import sys
import glob
import numpy as np
import trimesh

MODELS = sorted(glob.glob(os.path.join("real", "abo", "models", "*.glb")))


def probe(path):
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n{'=' * 70}\n{name}  ({size_mb:.1f} MB)")
    try:
        scene = trimesh.load(path, force="scene")
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return
    nodes = list(scene.graph.nodes_geometry)
    print(f"  L1 nodes/primitives: {len(nodes)}")
    total_faces = 0
    geoms = []
    for node in nodes:
        T, gname = scene.graph[node]
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        mat = g.visual.material
        mname = getattr(mat, "name", None) or type(g.visual).__name__
        print(f"    node={node!r} geom={gname!r} faces={len(g.faces)} material={mname!r}")
        geoms.append((node, g))
        total_faces += len(g.faces)
    print(f"  total faces: {total_faces}")

    # L2 connected components per node
    any_big_split = False
    for node, g in geoms:
        try:
            comps = g.split(only_watertight=False)
        except Exception as e:
            print(f"    {node}: split failed {e}")
            continue
        if len(comps) <= 1:
            print(f"    {node}: 1 connected shell")
            continue
        comps = sorted(comps, key=lambda c: -len(c.faces))
        shares = [len(c.faces) / max(total_faces, 1) for c in comps[:8]]
        big = [s for s in shares if s >= 0.03]
        print(f"    {node}: {len(comps)} shells; top face-shares "
              f"{[f'{s:.1%}' for s in shares[:8]]} "
              f"-> parts>=3%: {len(big)}")
        if len(big) >= 2:
            any_big_split = True
    verdict = ("AUTO-SPLIT PROMISING (>=2 sizable shells)" if any_big_split else
               ("primitive-level split only" if len(geoms) > 1 else "SINGLE SHELL, SINGLE MATERIAL"))
    print(f"  VERDICT: {verdict}")


for p in MODELS:
    probe(p)
print("\ndone")
