"""Color balance using compositional data methods on Gifti surface mesh."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tetrahydra.core as tet
from matplotlib.colors import LogNorm
from tetrahydra.utils import truncate_range, scale_range
from nibabel import save
from nibabel.gifti.gifti import GiftiImage, GiftiDataArray
from scipy.io import savemat

# Load maps
metric1 = load('/home/faruk/Git/parcellation/sample_data/01/rh.R1.shape.gii')
metric2 = load('/home/faruk/Git/parcellation/sample_data/01/rh.T2.shape.gii')
metric3 = load('/home/faruk/Git/parcellation/sample_data/01/rh.CBF.shape.gii')

# Load surface mesh
surf = load('/home/faruk/Git/parcellation/sample_data/rh.inflated.surf.gii')
nr_verts = surf.darrays[0].data.shape[0]

# Load vertex mask
mask = load('/home/faruk/Git/parcellation/sample_data/rh.mask.shape.gii')
mask = mask.darrays[0].data
mask = mask.astype(bool)

# Extract data
map_1 = np.squeeze(metric1.darrays[0].data)
map_2 = np.squeeze(metric2.darrays[0].data)
map_3 = np.squeeze(metric3.darrays[0].data)

# Special treatment for CBV (translate negative values)
map_3 = truncate_range(map_3, percMin=1, percMax=100)
map_3 = scale_range(map_3, scale_factor=500)

# Add empty CBF values to the mask
mask = mask * (map_3 > 0.1)

# Construct compositions
comp = np.zeros((np.sum(mask), 3))
comp[:, 0] = map_1[mask]
comp[:, 1] = map_2[mask]
comp[:, 2] = map_3[mask]

# Normalize dynamic ranges
comp = comp / np.max(comp, axis=0) * 500

# Truncate below zero and impute zero
comp[comp <= 0] = 0.001

# Lightness
light = (np.max(comp, axis=1) + np.min(comp, axis=1)) / 2.

# Closure
comp = tet.closure(comp)

# Hold original ilr coordinates for plotting
ilr_orig = tet.ilr_transformation(comp)

# Centering
center = tet.sample_center(comp)
temp = np.ones(comp.shape) * center
comp = tet.perturb(comp, temp**-1.)
# Standardize
totvar = tet.sample_total_variance(comp, center)
comp = tet.power(comp, np.power(totvar, -1./2.))

# Use Aitchison norm and powerinf for truncation of extreme compositions
anorm_thr = 1
anorm = tet.aitchison_norm(comp)
idx_trunc = anorm > anorm_thr
truncation_power = anorm[idx_trunc] / anorm_thr
correction = np.ones(anorm.shape)
correction[idx_trunc] = truncation_power
comp_bal = tet.power(comp, correction[:, None])

# Centered ilr coords
ilr = tet.ilr_transformation(comp_bal)

# Lightness balance
light = truncate_range(light, percMin=1, percMax=99)
hexc = comp_bal * light[:, None]

# Prepare gifti data
rgb_verts = np.zeros((nr_verts, 4))
rgb_verts[mask, 0] = hexc[:, 0]
rgb_verts[mask, 1] = hexc[:, 1]
rgb_verts[mask, 2] = hexc[:, 2]

# Save RGBA gifti -------------------------------------------------------------
#
# # Create gifti data arrays
# img = GiftiImage()
# darray_0 = GiftiDataArray(rgb_verts[:, 0])
# darray_1 = GiftiDataArray(rgb_verts[:, 1])
# darray_2 = GiftiDataArray(rgb_verts[:, 2])
# img.add_gifti_data_array(darray_0)
# img.add_gifti_data_array(darray_1)
# img.add_gifti_data_array(darray_2)
#
# # TODO: I could not get connectome workbench to load this data
# save(img, '/home/faruk/Git/parcellation/sample_data/derived/rh.fused.rgba.gii')

# Convert to BV files for RGBA visualization ---------------------------------
mat = dict()
mat['vertices'] = surf.darrays[0].data
mat['faces'] = (surf.darrays[1].data + 1)[:, ::-1]  # for matlab indexing

# Dynamic range mapping for RGB
rgb_verts[:, 0] = scale_range(rgb_verts[:, 0], scale_factor=255)
rgb_verts[:, 1] = scale_range(rgb_verts[:, 1], scale_factor=255)
rgb_verts[:, 2] = scale_range(rgb_verts[:, 2], scale_factor=255)
rgb_verts[:, 3] = rgb_verts[:, 3] + 255
rgb_verts = rgb_verts.astype('uint8')
mat['rgba'] = rgb_verts

# Save mat file
savemat('/home/faruk/Git/parcellation/sample_data/derived/rh_rgba.mat', mat)

# Plots ----------------------------------------------------------------------
fig = plt.figure()
limits = [-4, 4]

# Plot 2D histogram of ilr transformed data
ax_1 = plt.subplot(121)
_, _, _, h_1 = ax_1.hist2d(ilr_orig[:, 0], ilr_orig[:, 1], bins=2000,
                           cmap='gray_r')
h_1.set_norm(LogNorm(vmax=np.power(10, 2)))
plt.colorbar(h_1)
ax_1.set_title('Before Centering')
ax_1.set_xlabel('$v_1$')
ax_1.set_ylabel('$v_2$')
ax_1.set_aspect('equal')
ax_1.set_xlim(limits)
ax_1.set_ylim(limits)

# Plot 2D histogram of ilr transformed data
ax_2 = plt.subplot(122)
_, _, _, h_2 = ax_2.hist2d(ilr[:, 0], ilr[:, 1], bins=2000, cmap='gray_r')
h_2.set_norm(LogNorm(vmax=np.power(10, 2)))
plt.colorbar(h_2)
ax_2.set_title('After Centering')
ax_2.set_xlabel('$v_1$')
ax_2.set_ylabel('$v_2$')
ax_2.set_aspect('equal')
ax_2.set_xlim(limits)
ax_2.set_ylim(limits)

# plot axes of primary colors on top
c_axes = np.array([[15., 1., 1.], [1., 15., 1.], [1., 1., 15.]])
c_axes = tet.closure(c_axes)
c_axes = tet.ilr_transformation(c_axes)
caxw = 0.025  # width
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[0, 0], c_axes[0, 1], width=caxw,
                                  facecolor='r', edgecolor='none'))
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[1, 0], c_axes[1, 1], width=caxw,
                                  facecolor='g', edgecolor='none'))
ax_1.add_patch(patches.FancyArrow(0, 0, c_axes[2, 0], c_axes[2, 1], width=caxw,
                                  facecolor='b', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[0, 0], c_axes[0, 1], width=caxw,
                                  facecolor='r', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[1, 0], c_axes[1, 1], width=caxw,
                                  facecolor='g', edgecolor='none'))
ax_2.add_patch(patches.FancyArrow(0, 0, c_axes[2, 0], c_axes[2, 1], width=caxw,
                                  facecolor='b', edgecolor='none'))

plt.show()
