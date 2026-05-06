import numpy as np
import matplotlib.pyplot as plt

def teichmuller_approx(z_target, grid_zeta, mu_zeta, dA):
    """Vectorized approximation of the Beltrami integration."""
    # Reshape for broadcasting: (zeta_points, target_points)
    # This can be memory intensive for very large grids!
    diff = grid_zeta[:, np.newaxis] - z_target[np.newaxis, :]
    
    # Handle the singularity where zeta == z
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_diff = 1.0 / diff
        inv_diff[np.abs(diff) < 1e-10] = 0
        
    integral = np.sum(mu_zeta[:, np.newaxis] * inv_diff, axis=0) * dA
    return z_target - (1/np.pi) * integral

# 1. Setup the 'Source' Grid (Zeta) where the distortion lives
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)
X, Y = np.meshgrid(x, y)
grid_zeta = (X + 1j*Y).flatten()
dA = (x[1] - x[0]) * (y[1] - y[0])

# 2. Define mu(zeta): A localized 'bump' of distortion
# mu must satisfy |mu| < 1. We'll put a bump at 0.5 + 0.5j
dist = np.abs(grid_zeta - (0.5 + 0.5j))
mu_zeta = 0.5 * np.exp(-dist**2 / 0.2) 

# 3. Define the 'Target' points (the grid we want to warp)
# We use a slightly smaller grid to avoid edge artifacts
tx = np.linspace(-1, 1, 20)
ty = np.linspace(-1, 1, 20)
TX, TY = np.meshgrid(tx, ty)
z_target = (TX + 1j*TY).flatten()

# 4. Apply the mapping
z_warped = teichmuller_approx(z_target, grid_zeta, mu_zeta, dA)

# 5. Visualize the result
plt.figure(figsize=(10, 5))

# Plot Original Grid
plt.subplot(1, 2, 1)
plt.scatter(z_target.real, z_target.imag, s=10, c='blue', alpha=0.5)
plt.title("Original Conformal Grid")
plt.axis('equal')

# Plot Warped Grid
plt.subplot(1, 2, 2)
plt.scatter(z_warped.real, z_warped.imag, s=10, c='red')
plt.title("Warped by $\mu(z)$ (Teichmüller)")
plt.axis('equal')

plt.tight_layout()
plt.show()
