#!/usr/bin/env python
# coding: utf-8

# monte carlo method to plot the limit set of a quasi-fuchsian group

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import schottky_tools as st
import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import aggregate, rasterize
import holoviews as hv
import bokeh
from bokeh.resources import INLINE
import bokeh.io
import gc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datashader.transfer_functions as tf
from PIL import Image
import holoviews as hv
import pandas as pd
import random
hv.extension('bokeh')

# Set plot style to dark
plt.style.use('dark_background')

# 3. Optional: Boost performance for your limit sets
# This tells Bokeh to use the GPU (WebGL) for drawing if available
hv.renderer('bokeh').webgl = True

# 1. Define the inverse mapping to prevent backtracking
inverses = {'a': 'A', 'A': 'a', 'b': 'B', 'B': 'b'}


def get_monte_carlo_schottky(gen_maps, num_points=5_000_000, burn_in=500):
    # Start at any center
    current_pt = centers[0]
    points = np.zeros((num_points, 2))
    
    # List of map names for random selection
    names = list(gen_maps.keys())
    
    # 1. Burn-in phase (to reach the attractor)
    for _ in range(burn_in):
        f = gen_maps[random.choice(names)]
        current_pt = f(current_pt)
    
    # 2. Collection phase
    for i in range(num_points):
        # Pick a random generator
        # Note: In some groups, you should avoid picking the inverse i
        # of the last move, but for random iteration, the attractor 
        # usually handles this naturally.
        f = gen_maps[random.choice(names)]
        current_pt = f(current_pt)
        
        points[i, 0] = current_pt.real
        points[i, 1] = current_pt.imag
        
    return points


# Execute the DFS plot
# Level 5-6 is usually plenty for a detailed visual without crashing the kernel

#gap, r = 0, 1
#r0, r1 = r, r
#r2, r4 = r, r

# notice this gives not connected limit set 
# centers = [1+1j, -1-1j, 1-1j, -1+1j]
#centers = [1+1j, 1-1j, -1-1j, -1+1j]


# --- Quasi-Fuchsian Kissing Setup ---
r0 = 2.1
r1 = 1.04865589736  
r2 = r0
r3 = r1

# These centers are specifically chosen to be "offset"
# so they don't form a simple symmetric Fuchsian circle.
centers = [
    -1.6 - 2.5j,    # 0: Source a
    -0.885 + 0.5664j, # 2: Source b
     1.6 + 2.5j,    # 1: Target a
     0.885 - 0.5664j  # 3: Target b
]

c0 = centers[0]
c2 = centers[2]


radii = [r0, r1, r2, r3] 

# compare first radius to its matching radius
if r0 != r2:
    p, q, r, s = c0, c2, r0, r2
    C = p + (r*(q-p))/(r-s)
    R = np.sqrt((r*s*np.abs(q-p)**2)/((r-s)**2) - r*s)
    mid_inv = st.reflection_about(C, R)
else:
    mid_inv = st.midline_reflection(c0, c2)

basic_inv = st.simple_inversion(c0, r0)
map_a = st.compose(mid_inv, basic_inv)
map_a_inv = st.compose(basic_inv, mid_inv)

# again
c1 = centers[1]
c3 = centers[3]


# Use r1 consistently so you don't rely on the 'r' defined in the if-block
if r1 != r3:
    p, q, r, s = c1, c3, r1, r3
    C = p + (r*(q-p))/(r-s)
    R = np.sqrt((r*s*np.abs(q-p)**2)/((r-s)**2) - r*s)
    mid_inv = st.reflection_about(C, R)
else:
    mid_inv = st.midline_reflection(c1, c3)

basic_inv = st.simple_inversion(c1, r3)
map_b = st.compose(mid_inv, basic_inv)
map_b_inv = st.compose(basic_inv, mid_inv)

map_a = st.get_hyperbolic_transformation(-0.5 + 0j, 0 - 0.5j)
map_a_inv = st.get_hyperbolic_transformation(0 - 0.5j, -0.5)
map_b = st.get_hyperbolic_transformation(-0.5 + 0j, 0 - 0.5j)
map_b_inv = st.get_hyperbolic_transformation(-0.5j, - 0.5)

# define a quasi-conformal map
def f(s):
	def map_func(z):
		# this one is just affine
		# return z + k*np.conj(s)
		return z*np.abs(z)**s
	return map_func

# define its inverse
def F(s):
	"""
	Inverse of f(z) = z * |z|^s
	Formula: f_inv(w) = w * |w|**(-s / (s + 1))
	"""
	# We use np.power to handle arrays efficiently
	# Note: If s = -1, the original map is not invertible (collapses to a point/infinity)
	
	exponent = -s / (s + 1)
	
	def map_func(w):
		mag_w = np.abs(w)
		return w * np.power(mag_w, exponent)
	return map_func

s = 1.5

g = f(s)
G = F(s)

from scipy.interpolate import RegularGridInterpolator

# 1. Setup the 'Source' Grid (Zeta) where the distortion lives
xs = np.linspace(-2, 2, 100)
ys = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(xs, ys)
grid_zeta = (X + 1j*Y).flatten()
dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
dist = np.abs(grid_zeta - (0.5 + 0.5j))

# 2. Define mu(zeta): A localized 'bump' of distortion
# mu must satisfy |mu| < 1. We'll put a bump at 0.5 + 0.5j
# mu_zeta = 0.5 * np.exp(-dist**2 / 0.2) 
# Create a 'Ring' of distortion instead of a single bump
# A complex mu introduces rotation, which creates deeper 'wiggles'
phase = np.exp(1j * np.angle(grid_zeta) * 3) # Triple-pulsating phase
mu_zeta = 0.6 * phase * np.exp(-(np.abs(grid_zeta) - 0.8)**2 / 0.1)

# Create the coarse target grid for the mapping
res = 20
x = np.linspace(-2, 2, res)
y = np.linspace(-2, 2, res)

# Use indexing='ij' to make X correspond to x-axis and Y to y-axis
# This prevents the dimension mismatch in the interpolator!
X, Y = np.meshgrid(x, y, indexing='ij')
grid_z = X + 1j*Y

# Compute the warped positions
warped_grid_flat = st.teichmuller_approx(grid_z.flatten(), grid_zeta, mu_zeta, dA)

# Reshape to (res, res) to match the 'ij' meshgrid
warped_grid = warped_grid_flat.reshape((res, res))

# Define the interpolators
from scipy.interpolate import RegularGridInterpolator

# Now Dimension 0 is x (length 20) and Dimension 1 is y (length 20)
phi_real = RegularGridInterpolator((x, y), warped_grid.real, 
                                    bounds_error=False, fill_value=None)
phi_imag = RegularGridInterpolator((x, y), warped_grid.imag, 
                                    bounds_error=False, fill_value=None)



def Phi(z):
    p = np.array([[z.real, z.imag]])
    return phi_real(p)[0] + 1j * phi_imag(p)[0]

# Create the Inverse Map: Phi_inv(z) 
# (By mapping warped_grid back to the original grid_z)
# This is used for the conjugation: f_new = Phi * f * Phi_inv
# ... [Same logic as above but swap X/Y with warped_grid.real/.imag]

from scipy.interpolate import LinearNDInterpolator

# 1. Prepare the "Scattered" source points (where they ended up)
# These are the warped coordinates
warped_points_coords = np.column_stack((warped_grid.real.flatten(), 
                                        warped_grid.imag.flatten()))

# 2. Prepare the "Target" values (where they came from)
# This is the original structured grid_z
original_x_values = X.flatten()
original_y_values = Y.flatten()

# 3. Create the Inverse Interpolators
# We map warped -> original
# LinearNDInterpolator is better for "scattered" data than RegularGrid
phi_inv_real = LinearNDInterpolator(warped_points_coords, original_x_values)
phi_inv_imag = LinearNDInterpolator(warped_points_coords, original_y_values)

def Phi_inv(z):
    """Maps warped points back to the Fuchsian plane."""
    p = np.array([[z.real, z.imag]])
    # Handle potential NaNs if a point falls outside the convex hull
    res_x = phi_inv_real(p)[0]
    res_y = phi_inv_imag(p)[0]
    
    # Fallback: if NaN, return the original point (identity map)
    if np.isnan(res_x): 
        return z
    return res_x + 1j * res_y

g = Phi
G = Phi_inv

map_a = st.compose(g,st.compose(map_a,G))
map_a_inv = st.compose(g,st.compose(map_a_inv,G))
map_b = st.compose(g,st.compose(map_b,G))
map_b_inv = st.compose(g,st.compose(map_b_inv,G))



gen_map_f = {
    'a': map_a,
    'A': map_a_inv,
    'b': map_b,
    'B': map_b_inv
}

# - - - - - - - - - - - - - - - #
# Size 
# - - - - - - - - - - - - - - - #

# fig size
fig_size_inches = 12  # This will give us a 6000x6000 pixel image at 300 DPI
# - - - - - - - - - - - - - - - #
# DPI data per inch 
# - - - - - - - - - - - - - - - #
dpi = 300
# - - - - - - - - - - - - - - - #
#  Canvas
# - - - - - - - - - - - - - - - #
canvas_hw = 1200  # 12k x 12k pixels
# - - - - - - - - - - - - - - - #
# Execute
num_points = 50_000
burn_in = 500  
# - - - - - - - - - - - - - - - #

mc_points = st.get_monte_carlo_schottky(gen_map_f, centers, num_points, burn_in)

# mc_points = Phi(mc_points)
#quasi_x = phi_real(mc_points)
#quasi_y = phi_imag(mc_points)
#
#
#quasi_points = np.column_stack((quasi_x, quasi_y))
# Now create the DataFrame for Datashader
# df = pd.DataFrame(quasi_points, columns=['x', 'y'])

df = pd.DataFrame(mc_points, columns=['x', 'y'])




# Check for Infinite values specifically
print(f"\nNumber of Inf values: {np.isinf(df.to_numpy()).sum()}")



# 1. Calculate a tight bounding box using percentiles
# This ignores outliers and focuses on where 98% of your limit set lives
x_min, x_max = df.x.quantile([0.01, 0.99])
y_min, y_max = df.x.quantile([0.01, 0.99])

# 2. Add a small 'buffer' (e.g. 5%) so the fractal isn't touching the edge
x_range = x_max - x_min
y_range = y_max - y_min

padding = 0.05
tight_x = (x_min - x_range*padding, x_max + x_range*padding)
tight_y = (y_min - y_range*padding, y_max + y_range*padding)


cvs = ds.Canvas(plot_width=canvas_hw, plot_height=canvas_hw, 
                x_range=tight_x, y_range=tight_y)

# Clear the previous aggregator if it exists to free up RAM
if 'agg' in locals():
    del agg
gc.collect()

agg = cvs.points(df, 'x', 'y')

print(f"New Bounding Box: X{tight_x}, Y{tight_y}")
# 2. Manual Render at High DPI
# We set the figure size to match the aspect ratio
data = agg.values

fig = plt.figure(figsize=(fig_size_inches,fig_size_inches), facecolor='black')
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# Use LogNorm to ensure the "Cantor dust" of the limit set is visible
# vmin=1 ignores the empty black pixels (0 density)

# Pass everything to the constructor
from matplotlib.colors import PowerNorm

# gamma=0.3 is usually great for "whispy" limit sets
my_norm = PowerNorm(gamma=0.3, vmin=1, vmax=data.max())
# my_norm = LogNorm(vmin=1, vmax=data.max()) 

# Then just pass the norm object to imshow
im = ax.imshow(data, norm=my_norm, cmap='magma', origin='lower')

# 3. Save as a massive PNG
# dpi=300 at 20x20 inches is 6000x6000 pixels
export_name = f"schottky_mc_{fig_size_inches}x{fig_size_inches}_{canvas_hw}_pixels_{num_points}_points_{dpi}_dpi.png"

plt.savefig(export_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"High-res file saved: {export_name}")




