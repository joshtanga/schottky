import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets, interact
from IPython.display import display, clear_output

# --- 1. Math Core ---
def get_schottky_matrix(c1, c2, r):
    return np.array([[c2/r, (r**2 - c1*c2)/r], [1/r, -c1/r]], dtype=complex)

def mobius(m, z):
    denom = m[1,0]*z + m[1,1]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return (m[0,0]*z + m[0,1]) / denom

# --- 2. Setup Generators ---
gap, r = 0.35, 0.45
centers = [np.exp(1j * (0 + gap)), np.exp(1j * (np.pi - gap)),
           np.exp(1j * (np.pi/2 + gap)), np.exp(1j * (3*np.pi/2 - gap))]

# Mapping: 'a' maps 0->1, 'A' maps 1->0, 'b' maps 2->3, 'B' maps 3->2
mat_a = get_schottky_matrix(centers[0], centers[1], r)
mat_b = get_schottky_matrix(centers[2], centers[3], r)
mat_A = np.linalg.inv(mat_a)
mat_B = np.linalg.inv(mat_b)

gen_map = {'a': mat_a, 'A': mat_A, 'b': mat_b, 'B': mat_B}
colors = ['red', 'red', 'green', 'green']

theta = np.linspace(0, 2*np.pi, 500)
base_circle = np.exp(1j*theta) * r
initial_circles = [base_circle + c for c in centers]

# --- 3. Interactive UI ---
text_input = widgets.Text(
    value='',
    placeholder="Type 'a', 'A', 'b', or 'B'...",
    description='Word:',
    disabled=False
)

out = widgets.Output()

def update_plot(change):
    word = change['new']
    with out:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Plot the initial 4 circles (Fundamental Domain)
        for i, z in enumerate(initial_circles):
            ax.plot(z.real, z.imag, color=colors[i], lw=2, alpha=0.3, label="Original" if i==0 else "")
        
        # 2. Compute the cumulative matrix from the string
        # Example: 'ab' means apply 'a' then 'b'
        current_mat = np.eye(2, dtype=complex)
        valid_word = True
        
        for char in word:
            if char in gen_map:
                current_mat = gen_map[char] @ current_mat
            else:
                valid_word = False
        
        # 3. Transform and plot the new generation
        if word != "":
            for i, z_init in enumerate(initial_circles):
                z_new = mobius(current_mat, z_init)
                ax.plot(z_new.real, z_new.imag, color=colors[i], lw=2)
        
        ax.set_aspect('equal')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f"Group Action for word: {word}")
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if not valid_word:
            plt.text(-1.8, -1.8, "Invalid chars! Use a, A, b, B", color='red')
            
        plt.show()

# Observe changes in the text box
text_input.observe(update_plot, names='value')

display(text_input, out)

# Trigger initial plot
update_plot({'new': ''})
