import numpy as np
import plotly.graph_objects as go

# --- 1. Math Core ---
def get_schottky_matrix(c1, c2, r):
    return np.array([[c2/r, (r**2 - c1*c2)/r], [1/r, -c1/r]], dtype=complex)

def mobius(m, z):
    denom = m[1,0]*z + m[1,1]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return (m[0,0]*z + m[0,1]) / denom

# --- 2. Configuration ---
gap, r = 0.01, 0.68
centers = [np.exp(1j * (0 + gap)), np.exp(1j * (np.pi - gap)),
           np.exp(1j * (np.pi/2 + gap)), np.exp(1j * (3*np.pi/2 - gap))]
mat_a = get_schottky_matrix(centers[0], centers[1], r)
mat_b = get_schottky_matrix(centers[2], centers[3], r)
gens = [mat_a, mat_b, np.linalg.inv(mat_a), np.linalg.inv(mat_b)]
colors = ['red', 'red', 'green', 'green']
theta = np.linspace(0, 2*np.pi, 100) # Lower resolution for faster scrubbing
base_circle = np.exp(1j*theta) * r

# --- 3. Recursive Generation ---
def get_disks_at_depth(current_mat, current_depth, target_depth, last_gen_idx=None):
    if current_depth == target_depth:
        return [(mobius(current_mat, base_circle + c), colors[i]) for i, c in enumerate(centers)]
    found = []
    for i, g in enumerate(gens):
        if last_gen_idx is not None and i == (last_gen_idx + 2) % 4: continue
        found.extend(get_disks_at_depth(current_mat @ g, current_depth + 1, target_depth, i))
    return found

# --- 4. Assemble Plotly Figure ---
fig = go.Figure()
max_depth = 4
trace_indices_by_depth = {}
current_idx = 0

# We pre-add every single circle to the plot
for d in range(max_depth + 1):
    trace_indices_by_depth[d] = []
    if d == 0:
        for i, c in enumerate(centers):
            z = base_circle + c
            fig.add_trace(go.Scatter(x=z.real, y=z.imag, mode='lines',
                         line=dict(color=colors[i], width=2),
                         visible=True, showlegend=False)) # Only depth 0 visible at start
            trace_indices_by_depth[d].append(current_idx)
            current_idx += 1
    else:
        disks = get_disks_at_depth(np.eye(2, dtype=complex), 0, d)
        for z, color in disks:
            fig.add_trace(go.Scatter(x=z.real, y=z.imag, mode='lines',
                         line=dict(color=color, width=0.8),
                         visible=False, showlegend=False))
            trace_indices_by_depth[d].append(current_idx)
            current_idx += 1

# --- 5. Create the Visibility Slider ---
steps = []
for d in range(max_depth + 1):
    # Create a list of booleans: True if trace depth <= d, else False
    visibility = [False] * current_idx
    for active_depth in range(d + 1):
        for idx in trace_indices_by_depth[active_depth]:
            visibility[idx] = True
            
    step = dict(
        method="restyle",
        args=["visible", visibility],
        label=f"Depth {d}"
    )
    steps.append(step)

fig.update_layout(
    sliders=[dict(active=0, currentvalue={"prefix": "Iteration: "}, steps=steps)],
    xaxis=dict(range=[-2, 2], scaleanchor="y", scaleratio=1, zeroline=False),
    yaxis=dict(range=[-2, 2], zeroline=False),
    title="Schottky Iteration Scrubber",
    width=800, height=800,
    template="plotly_white"
)

fig.show()
