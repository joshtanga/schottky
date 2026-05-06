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

