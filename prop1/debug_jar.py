import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiments import *

objs = [load(os.path.join(HERE, 'data', f)) for f in sorted(os.listdir(os.path.join(HERE, 'data'))) if f.endswith('.npz')]
o = [x for x in objs if x['name'] == 'jar_lid_00.npz'][0]
masks = part_masks(o['label'], o['parts'])
ref_skel = skeletonize(o['occ'])
lbl = assign_labels(ref_skel, masks)
print('skeleton voxels:', int(ref_skel.sum()))
print('label counts:', {int(v): int(c) for v, c in zip(*np.unique(lbl, return_counts=True))})
print('GT edges:', o['gt'])
pred = pair_paths(ref_skel, lbl, 2, set())
print('C0 predicted pairs:', pred)
inside = ref_skel & ~(masks[0] | masks[1])
print('skel voxels outside both parts:', int(inside.sum()))
dnear = np.stack([ndimage.distance_transform_edt(~m) for m in masks]).min(axis=0)
print('max nearest-part dist on skeleton:', float(dnear[ref_skel].max()))
sub = ref_skel & ((lbl == 1) | (lbl == 2) | (lbl == 0))
cc, n = ndimage.label(sub, S26)
print('sub components:', n)
print('cc ids touching part1:', np.unique(cc[lbl == 1]))
print('cc ids touching part2:', np.unique(cc[lbl == 2]))
