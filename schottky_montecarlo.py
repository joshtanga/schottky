#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import datashader.transfer_functions as tf
from PIL import Image


import holoviews as hv
hv.extension('bokeh')

# Set plot style to dark
plt.style.use('dark_background')

# 3. Optional: Boost performance for your limit sets
# This tells Bokeh to use the GPU (WebGL) for drawing if available
hv.renderer('bokeh').webgl = True


# 1. Define the inverse mapping to prevent backtracking
inverses = {'a': 'A', 'A': 'a', 'b': 'B', 'B': 'b'}


import numpy as np
import pandas as pd

import random

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


gen_map_f = {
    'a': map_a,
    'A': map_a_inv,
    'b': map_b,
    'B': map_b_inv
}

# Execute
mc_points = st.get_monte_carlo_schottky(gen_map_f, centers, num_points=500_000_000, burn_in = 1000)
df = pd.DataFrame(mc_points, columns=['x', 'y'])


# Check for Infinite values specifically
print(f"\nNumber of Inf values: {np.isinf(df.to_numpy()).sum()}")



# 1. Calculate a tight bounding box using percentiles
# This ignores outliers and focuses on where 98% of your limit set lives
x_min, x_max = df.x.quantile([0.01, 0.99])
y_min, y_max = df.y.quantile([0.01, 0.99])

# 2. Add a small 'buffer' (e.g. 5%) so the fractal isn't touching the edge
x_range = x_max - x_min
y_range = y_max - y_min

padding = 0.05
tight_x = (x_min - x_range*padding, x_max + x_range*padding)
tight_y = (y_min - y_range*padding, y_max + y_range*padding)

# - - - - - - - - - - - - - - - #
#  Canvas
# - - - - - - - - - - - - - - - #
cvs = ds.Canvas(plot_width=12000, plot_height=12000, 
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

# - - - - - - - - - - - - - - - #
# Size 
# - - - - - - - - - - - - - - - #

fig = plt.figure(figsize=(20, 20), facecolor='black')
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# Use LogNorm to ensure the "Cantor dust" of the limit set is visible
# vmin=1 ignores the empty black pixels (0 density)

# Pass everything to the constructor
my_norm = LogNorm(vmin=1, vmax=data.max()) 

# Then just pass the norm object to imshow
im = ax.imshow(data, norm=my_norm, cmap='magma', origin='lower')

# 3. Save as a massive PNG
# dpi=300 at 20x20 inches is 6000x6000 pixels
export_name = "schottky_ultra_res_mc_large.png"

# - - - - - - - - - - - - - - - #
# DPI data per inch 
# - - - - - - - - - - - - - - - #


plt.savefig(export_name, dpi=600, bbox_inches='tight', pad_inches=0)
plt.close(fig)

print(f"High-res file saved: {export_name}")




