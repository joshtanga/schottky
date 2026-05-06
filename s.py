import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display, clear_output

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


# compose midline inversion with the simple inversion to get a new transformation
def compose(f, g):
    def composed(z):
        return f(g(z))
    return composed




gap, r = 0, 1
r1, r2 = r, r

# notice this gives not connected limit set 
# centers = [1+1j, -1-1j, 1-1j, -1+1j]
centers = [1+1j, 1-1j, -1-1j, -1+1j]


# --- 2. Setup ---
# Radii for the two pairs
#r1, r2 = 2.1, 1.70110510247
radii = [r1, r1, r2, r2] # r1 for pair A, r2 for pair B

# Flatten centers into a single list of 4 complex numbers
# Pair A: centers[0] -> centers[1]
# Pair B: centers[2] -> centers[3]
#centers = [
#    1.6 + 2.5j,   # 0: Source a
#    -1.6 - 2.5j,  # 1: Target a
#    -2 + 1.28j,   # 2: Source b
#    2 - 1.28j     # 3: Target b
#]

# ID: 0=Red, 1=Orange, 2=Green, 3=Blue
bright_colors = ['#FF0000', '#FF8C00', '#008000', '#0000FF']
mapped_bright_colors = [bright_colors[1], bright_colors[0], bright_colors[3], bright_colors[2]]


# --- Quasi-Fuchsian Kissing Setup ---
# Pair A uses r1, Pair B uses r2
#r1 = 2.1
#r2 = 1.04865589736  

#radii = [r1, r1, r2, r2]

# These centers are specifically chosen to be "offset"
# so they don't form a simple symmetric Fuchsian circle.
#centers = [
#    -1.6 - 2.5j,    # 0: Source a
#     1.6 + 2.5j,    # 1: Target a
#    -0.885 + 0.5664j, # 2: Source b
#     0.885 - 0.5664j  # 3: Target b
#]

p = centers[0]
q = centers[2]
r = r1
s = r1

# Get the midline inversion function
C = p + (r*(q-p))/(r-s)
R = np.sqrt((r*s*np.abs(q-p)**2)/((r-s)**2) - r*s)

mid_inv = reflection_about(C,R)
basic_inv = simple_inversion(p, r1)
schottky_map_1 = compose(mid_inv, basic_inv)


