# --- this module includes ---
#
# relection_about(c, r): returns a function that performs reflection about a center and radius
# midline_reflection(c1, c2): returns a function that reflects points across the perpendicular bisector of c1 and c2
# midline_inversion(c1, c2, r): returns a function that performs midline inversion defined by two circles
# simple_inversion(c, r): returns a function that performs simple inversion about a circle defined by center c and radius r
# identity(z): the identity transformation
# compose(f, g): returns a function that composes two transformations f and g
# get_generator(c_pair, r_pair): computes the generator map and its inverse for a given pair of circles and their radii
# get_monte_carlo_schottky(gen_maps, centers, num_points=5_000_000, burn_in=500): generates points in the limit set of a Schottky group using a Monte Carlo method

import numpy as np

# define a quasi-conformal map
def f(A=1.4,k=.3):
    def map_func(z):
        return A*z + k
    return map_func

# define its inverse
def F(A=1.4,k=.3):
    def map_func(z):
        return (z-k)/A
    return map_func

# the Teichmüller approximation of a quasi-conformal map given its Beltrami coefficient mu defined on a grid of points zeta, with area element dA.
def teichmuller_approx(z_target, grid_zeta, mu_zeta, dA):
    """
    z_target: The points we want to map (can be an array)
    grid_zeta: The mesh of points where mu is defined
    mu_zeta: The distortion values at those points
    area_element: The size of each grid cell (dA)
    
    Vectorized approximation of the Beltrami integration."""
    # Reshape for broadcasting: (zeta_points, target_points)
    # This can be memory intensive for very large grids!
    diff = grid_zeta[:, np.newaxis] - z_target[np.newaxis, :]
    
    # Handle the singularity where zeta == z
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_diff = 1.0 / diff
        inv_diff[np.abs(diff) < 1e-10] = 0
        
    integral = np.sum(mu_zeta[:, np.newaxis] * inv_diff, axis=0) * dA
    return z_target - (1/np.pi) * integral


def get_hyperbolic_transformation(point_a, point_b):
    """
    Returns a function that maps point_a to point_b in the Poincare Disk.
    Uses the composition of two reflections (Möbius transformation).
    Points should be complex numbers: x + iy
    """
    
    def reflect(z, target_to_origin):
        """Reflects z across the geodesic bisecting 0 and target_to_origin"""
        a = target_to_origin
        # The formula for a reflection (inversion) across the 
        # hyperbolic perpendicular bisector of [0, a]
        return (np.conj(z) - a) / (np.conj(a) * np.conj(z) - 1)

    def transform(z):
        # Step 1: Reflect A to the origin
        z_mid = reflect(z, point_a)
        # Step 2: Since A is now at origin, we need to move origin to B.
        # We use a second reflection to map 0 -> B.
        # Note: we must account for the orientation flip of the first reflection.
        
        # A simpler approach for the composition: 
        # T(z) = (z - a)/(conj(a)z - 1) maps a -> 0
        # S(z) = (z + b)/(conj(b)z + 1) maps 0 -> b
        # Let's use the standard Mobius form for the final result:
        
        # Map A to 0
        z_zero = (z - point_a) / (np.conj(point_a) * z - 1)
        # Map 0 to B
        z_final = (z_zero + point_b) / (np.conj(point_b) * z_zero + 1)
        
        return z_final

    return transform

# Example Usage:
A = 0.5 + 0j
B = 0.0 + 0.5j
T = get_hyperbolic_transformation(A, B)

print(f"Mapping A: {T(A)}") # Should be approx 0 + 0.5j

# returns a function that performs reflection about a center and radius
def reflection_about(c, r):
    """
    Returns a function that performs reflection about a center and radius.
    
    Parameters:
    c: complex - center of the circle defining the reflection
    r: float - radius of the circle defining the reflection
    
    Returns:
    A function that takes a numpy array of complex numbers and returns their images under the reflection.
    """
    
    def reflect(z):
        """Applies the reflection to an array of complex numbers."""
        z = np.asarray(z, dtype=complex)
        return c + (r**2 * (z - c)) / np.abs(z - c)**2

    return reflect

# numpy arrays of complex numbers and return the image of that
# array under a midline inversion defined by two circles.
#
# the first circle is the one being inverted, and the second circle defines the midline about which the inversion is performed.

def midline_reflection(c1, c2):
    """
    Returns a function that reflects points across the 
    perpendicular bisector of c1 and c2.
    """
    midpoint = (c1 + c2) / 2
    # The term (c2 - c1) / conj(c2 - c1) represents the 
    # rotation required after the conjugation.
    rotation = -(c2 - c1) / np.conj(c2 - c1)
    
    def reflect(z):
        z = np.asarray(z, dtype=complex)
        return midpoint + rotation * (np.conj(z) - np.conj(midpoint))
            
    return reflect


def midline_inversion(c1, c2, r):
    """
    Returns a function that performs midline inversion defined by two circles.
    
    Parameters:
    c1: complex - center of the first circle (the one being inverted)
    c2: complex - center of the second circle (the midline)
    r: float - radius of the first circle
    
    Returns:
    A function that takes a numpy array of complex numbers and returns their images under the midline inversion.
    """
    
    # Calculate the transformation matrix for the midline inversion
    M = np.array([[c2/r, (r**2 - c1*c2)/r], [1/r, -c1/r]], dtype=complex)

    def mid_inv(z):
        """Applies the transformation to an array of complex numbers."""
        z = np.asarray(z, dtype=complex)
        # CONJUGATE z here to make this an orientation-reversing inversion
        z_conj = np.conj(z)
    
        numerator = M[0, 0] * z_conj + M[0, 1]
        denominator = M[1, 0] * z_conj + M[1, 1]
        return numerator / denominator
    
    return mid_inv 

# simple inverstion about the first circle
def simple_inversion(c, r):
    def inv(z):
        z = np.asarray(z, dtype=complex)
        return c + (r**2 * (z - c)) / np.abs(z - c)**2
    return inv


# the identity transformation
def identity(z):
    return z

# compose midline inversion with the simple inversion to get a new transformation
def compose(f, g):
    def composed(z):
        return f(g(z))
    return composed

# This function computes the generator map and its inverse for a given pair of circles and their radii.
def get_generator(c_pair, r_pair):
    p, q = c_pair
    r, s = r_pair

    if r != s:
        # Hyperbolic/Circular inversion logic
        C = p + (r * (q - p)) / (r - s)
        R = np.sqrt((r * s * np.abs(q - p)**2) / ((r - s)**2) - r * s)
        mid_inv = reflection_about(C, R)
    else:
        # Euclidean/Midline symmetry logic
        mid_inv = midline_reflection(p, q)

    basic_inv = simple_inversion(p, r)

    # Return both the map and its inverse
    return compose(mid_inv, basic_inv), compose(basic_inv, mid_inv)

import random
   



# This function generates points in the limit set of a Schottky group using a Monte Carlo method.    
def get_monte_carlo_schottky(gen_maps, centers, num_points=5_000_000, burn_in=500):
    current_pt = centers[0]
    points = np.zeros((num_points, 2))
    names = list(gen_maps.keys())

    for _ in range(burn_in):
        try:
            f = gen_maps[random.choice(names)]
            current_pt = f(current_pt)
        except ZeroDivisionError:
            current_pt = random.choice(centers)

    for i in range(num_points):
        try:
            f = gen_maps[random.choice(names)]
            next_pt = f(current_pt)

            if np.isfinite(next_pt):
                current_pt = next_pt
            else:
                current_pt = random.choice(centers)
                continue
        
        except ZeroDivisionError:
            current_pt = random.choice(centers)
            continue

        points[i, 0] = current_pt.real
        points[i, 1] = current_pt.imag

    return points
