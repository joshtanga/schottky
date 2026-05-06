%load_ext autoreload
%autoreload 2

# This code sets up a Schottky group with two generators (a and b) and their inverses (A and B).
# There is a fuchsian & quasi-fuchsian setup, where the circles are arranged in a "kissing" configuration.

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display, clear_output
import sys
import os
import schottky_tools as st

# --- static styling ---
# ID: 0=Red, 1=Orange, 2=Green, 3=Blue
bright_colors = ['#FF0000', '#FF8C00', '#008000', '#0000FF']
mapped_bright_colors = bright_colors


# --- choose fuchsian or quasi-fuchsian via this conditional block ---
quasi_fuchsian = True

# --- gluing syntax for data: [source_a, source_b, target_a, target_b]
if not quasi_fuchsian:
    # --- Fuchsian Data ---
    r0, r1, r2, r3 = 1, 1, 1, 1
    centers = [1+1j, 1-1j, -1-1j, -1+1j]
else:   
    # --- Quasi-Fuchsian Kissing Setup --- (from Desmos)
    r0, r1, r2, r3 = 2.1, 1.04865589736, 2.1, 1.04865589736
    
    # These centers are specifically chosen to be "offset" in Desmos
    # so they don't form a simple symmetric Fuchsian circle.
    centers = [
        -1.6 - 2.5j,    # 0: Source a
        -0.885 + 0.5664j, # 2: Source b
         1.6 + 2.5j,    # 1: Target a
         0.885 - 0.5664j  # 3: Target b
    ]

# --- data for circles define the generators ---
# the schottky map is a composition of a midline inversion and a simple inversion, 
# so we need to get the midline inversion function first, which depends on the centers and radii of the circles
# the first pair of circles is defined by centers[0] and centers[1], and the second pair is defined by centers[2] and centers[3]
c0, c2 = centers[0], centers[2]
c1, c3 = centers[1], centers[3]
radii = [r0, r1, r2, r3] 

# Execute for both pairs
map_a, map_a_inv = st.get_generator((centers[0], centers[2]), (r0, r2))
map_b, map_b_inv = st.get_generator((centers[1], centers[3]), (r1, r3))


# dictionary to map from character to the corresponding generator function
gen_map_f = {
    'a': map_a,
    'A': map_a_inv,
    'b': map_b,
    'B': map_b_inv
}
    
# Create each circle using its specific radius and center
theta = np.linspace(0, 2*np.pi, 200)
fundamental_circles = [np.exp(1j*theta) * radii[i] + centers[i] for i in range(4)]


# Stores tuples of (circle_points, color_index)
circle_history = [] 

# interactive widgets for input and output
text_input = widgets.Text(description='Word:', placeholder='e.g. a, ab, ba')
out = widgets.Output()

# callback function to update the plot based on the input word
def update_plot(change):
    word = change['new']
    if not word:
        return
    
    with out:
        # Calculate mapping for the current word
        m = np.eye(2, dtype=complex)
        f = st.identity
        
        for char in word:
            if char in gen_map_f:
                f = st.compose(gen_map_f[char],f)
                
        # Transform all 4 circles and store with their respective colors
        for i in range(4):
            #transformed_c = mobius(m, fundamental_circles[i])
            transformed_c = f(fundamental_circles[i])
            circle_history.append((transformed_c, i))
        
        # Draw everything
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Draw Fundamental Circles (Level 0)
        for i, c in enumerate(fundamental_circles):
            ax.plot(c.real, c.imag, color=mapped_bright_colors[i], lw=3, zorder=5)
            ax.plot(c.real, c.imag, color='black', lw=1, zorder=6, alpha=0.3)
            
        # 2. Draw Accumulated History (Nested Levels)
        for circle_data, color_idx in circle_history:
            ax.plot(circle_data.real, circle_data.imag, 
                    color=bright_colors[color_idx], lw=0.8, alpha=0.8, zorder=2)
        
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axis('off')
        plt.show()

# Reset Utility
reset_button = widgets.Button(description="Clear History")
def reset_history(b):
    global circle_history
    circle_history = []
    text_input.value = ""
    with out:
        clear_output(wait=True)
        # Re-render just the fundamentals
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, c in enumerate(fundamental_circles):
            ax.plot(c.real, c.imag, color=bright_colors[i], lw=3)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.axis('off')
        plt.show()

reset_button.on_click(reset_history)

text_input.observe(update_plot, names='value')
display(widgets.HBox([text_input, reset_button]), out)

# Initial Render
with out:
    update_plot({'new': ''})
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, c in enumerate(fundamental_circles):
        ax.plot(c.real, c.imag, color=bright_colors[i], lw=3)
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.axis('off')
    plt.show()

