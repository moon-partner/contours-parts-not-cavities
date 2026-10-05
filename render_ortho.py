"""Orthographic three-view renderer (Blender, Workbench) with per-part alpha masks.

Run (Blender 4.x/5.x):
    blender --background --python render_ortho.py -- --out E:/Blender/real/rendered --name demo
    blender --background --python render_ortho.py -- --glb model.glb --out DIR --name mug

Modes (default: both):
    --masks      render EACH mesh object alone -> alpha PNG = perfect part mask
                 (3 views x #parts; downstream needs no color segmentation at all)
    --composite  render all parts colored (palette auto-assigned) for eyeballing

Fixes vs the original plan script (see PITFALLS / design review):
  * Workbench MATERIAL shading reads mat.diffuse_color (Viewport Display), NOT the
    Principled BSDF node -- we set diffuse_color explicitly.
  * render_aa = OFF -> mask edges are strictly binary (no FXAA color bleeding).
  * camera orientation verified: front looks +Y, side looks -X, top looks -Z.
  * real argv parsing after '--' (the original batch snippet was broken).
  * no --glb given -> builds a procedural two-part mug for smoke testing.

Outputs under {out}/{name}/:
    {name}_{view}_{partIndex}_{partName}.png   per-part alpha mask (RGBA)
    {name}_{view}.png                          colored composite (RGBA)
    {name}_meta.json                           parts order, views, ortho scale, resolution
"""
import bpy
import sys
import os
import json
import math
from mathutils import Vector

# ---------------------------------------------------------------- argv
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return default if name not in argv else argv[argv.index(name) + 1]


GLB = arg("--glb")
OUT = arg("--out", os.path.join("E:", os.sep, "Blender", "real", "rendered"))
NAME = arg("--name", "demo")
RES = int(arg("--resolution", "512"))
DO_MASKS = "--composite-only" not in argv
DO_COMP = "--masks-only" not in argv

PALETTE = [(0.90, 0.90, 0.92), (0.85, 0.15, 0.15), (0.15, 0.45, 0.85),
           (0.15, 0.70, 0.30), (0.95, 0.75, 0.10), (0.55, 0.25, 0.75),
           (0.95, 0.45, 0.10), (0.35, 0.35, 0.35)]

# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)

if GLB:
    bpy.ops.import_scene.gltf(filepath=GLB)
else:
    # procedural 2-part mug for smoke testing: body (cylinder) + handle (torus half)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=(0, 0, 0))
    bpy.context.active_object.name = "body"
    bpy.ops.mesh.primitive_torus_add(major_radius=0.30, minor_radius=0.07,
                                     location=(0.62, 0, 0),
                                     rotation=(math.radians(90), 0, 0))
    bpy.context.active_object.name = "handle"
    for ob, mname in zip([bpy.context.scene.objects["body"],
                          bpy.context.scene.objects["handle"]], ["bodyMat", "handleMat"]):
        mat = bpy.data.materials.new(mname)
        ob.data.materials.append(mat)

# --- automatic part split (no manual Blender work):
# glTF primitives/materials arrive as material SLOTS on one object; masks are per
# OBJECT, so separate multi-material objects by material first (scriptable, headless).
if GLB and "--no-sep" not in argv:
    for ob in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        if len(ob.data.materials) <= 1:
            continue
        n_slots = len([m for m in ob.data.materials if m])
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='MATERIAL')
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  split {ob.name!r}: {n_slots} material slots -> separate objects")
# rename single-material objects to their (sanitized) material name = part name
import re
for ob in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
    if len(ob.data.materials) == 1 and ob.data.materials[0]:
        raw = ob.data.materials[0].name.strip('"')
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", raw)[:40].strip("_")
        if clean:
            ob.name = clean

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if len(meshes) < 2:
    print(f"WARN: only {len(meshes)} mesh object(s). If body+handle are one object, "
          f"separate them in Edit Mode (P) first -- masks are per OBJECT.")
for i, ob in enumerate(meshes):
    ob.data.materials.clear()
    mat = bpy.data.materials.new(f"mat_{ob.name}")
    mat.diffuse_color = (*PALETTE[i % len(PALETTE)], 1.0)   # Workbench reads THIS
    ob.data.materials.append(mat)

# ---------------------------------------------------------------- bbox & camera
mins = Vector((1e9, 1e9, 1e9))
maxs = Vector((-1e9, -1e9, -1e9))
for ob in meshes:
    for c in ob.bound_box:
        w = ob.matrix_world @ Vector(c)
        mins = Vector(map(min, mins, w))
        maxs = Vector(map(max, maxs, w))
center = (mins + maxs) / 2
max_dim = max((maxs - mins))

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.resolution_x = RES
scene.render.resolution_y = RES
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True
scene.display.shading.light = 'FLAT'
scene.display.shading.color_type = 'MATERIAL'
scene.display.shading.show_object_outline = False
scene.display.shading.show_specular_highlight = False
scene.display.render_aa = 'OFF'          # binary mask edges (see header)
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")

cam_data = bpy.data.cameras.new("OrthoCam")
cam_data.type = 'ORTHO'
cam_data.ortho_scale = max_dim * 1.1
cam = bpy.data.objects.new("OrthoCam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

VIEWS = {
    "front": ((0, -1, 0), (math.radians(90), 0, 0)),   # camera at -Y, looks +Y
    "side":  ((1, 0, 0),  (math.radians(90), 0, math.radians(90))),  # at +X, looks -X
    "top":   ((0, 0, 1),  (0, 0, 0)),                  # at +Z, looks -Z
}

outdir = os.path.join(OUT, NAME)
os.makedirs(outdir, exist_ok=True)
meta = dict(name=NAME, resolution=RES, ortho_scale=round(cam_data.ortho_scale, 4),
            center=[round(v, 4) for v in center],
            views=list(VIEWS), parts=[o.name for o in meshes],
            source=GLB or "builtin-procedural-mug")


def render(path):
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"  -> {path}")


for view, (direction, rot) in VIEWS.items():
    cam.location = Vector(center) + Vector(direction) * max_dim * 3
    cam.rotation_euler = rot

    if DO_COMP:
        for ob in meshes:
            ob.hide_render = False
        render(os.path.join(outdir, f"{NAME}_{view}.png"))

    if DO_MASKS:
        for i, ob in enumerate(meshes):
            for other in meshes:
                other.hide_render = (other is not ob)
            render(os.path.join(outdir, f"{NAME}_{view}_{i}_{ob.name}.png"))
        for ob in meshes:
            ob.hide_render = False

with open(os.path.join(outdir, f"{NAME}_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print("DONE", json.dumps(meta))

# optional: export the scene as GLB so the GT pipeline (build_abo_gt.py) can consume
# the exact same geometry the masks were rendered from (smoke tests without ABO).
EXPORT = arg("--export-glb")
if EXPORT:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if GLB:
        bpy.ops.import_scene.gltf(filepath=GLB)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=(0, 0, 0))
        bpy.context.active_object.name = "body"
        bpy.ops.mesh.primitive_torus_add(major_radius=0.30, minor_radius=0.07,
                                         location=(0.62, 0, 0),
                                         rotation=(math.radians(90), 0, 0))
        bpy.context.active_object.name = "handle"
        for ob, mname in zip([bpy.context.scene.objects["body"],
                              bpy.context.scene.objects["handle"]],
                             ["bodyMat", "handleMat"]):
            ob.data.materials.append(bpy.data.materials.new(mname))
    os.makedirs(os.path.dirname(EXPORT) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=EXPORT, use_selection=False)
    print("EXPORTED", EXPORT)
