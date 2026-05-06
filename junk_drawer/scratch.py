# returns a function that performs reflection about a center and radius

import numpy as np

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

def midline_reflection(c1, c2):
    """
    Returns a function that reflects points across the 
    perpendicular bisector of c1 and c2.
    """
    midpoint = (c1 + c2) / 2
    # The term (c2 - c1) / conj(c2 - c1) represents the 
    # rotation required after the conjugation.
    rotation = (c2 - c1) / np.conj(c2 - c1)
    
    def reflect(z):
        z = np.asarray(z, dtype=complex)
        return midpoint + rotation * (np.conj(z) - np.conj(midpoint))
            
        return reflect

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

# Get the midline inversion function
C = p + (r*(q-p))/(r-s)
R = np.sqrt((r*s*np.abs(q-p)**2)/((r-s)**2) - r*s)

mid_inv = reflection_about(C,R)

# Test the midline inversion on a grid of points
theta = np.linspace(0, 2 * np.pi, 100)
circle1 = c1 + r1 * np.exp(1j * theta)  # Points on the first circle
circle2 = c2 + r2 * np.exp(1j * theta)  # Points on the second circle



basic_inv = simple_inversion(c1, r)
schottky_map = compose(mid_inv, basic_inv) 
img = schottky_map(schottky_map(my_shape))
#jcircle1 = basic_inv(my_shape)
#img = mid_inv(my_shape)



