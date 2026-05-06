# python module to return a mapping that can accept
# numpy arrays of complex numbers and return the image of that
# array under a midline inversion defined by two circles.
#
# the first circle is the one being inverted, and the second circle defines the midline about which the inversion is performed.

import math
import numpy as np

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

