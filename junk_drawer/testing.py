# test to see if various functions work as expected

import numpy as np
from midline_inversion import midline_inversion 
from reflection_about import reflection_about
import schottky_tools as st


# Define two circles for the midline inversion
c1 = 6 + 1j  # Center of the first circle
c2 = 0 + 0j  # Center of the second circle (midline)
r1 = 1        # Radius of the first circle   
r2 = 1        # Radius of the second circle

# Test the midline inversion on a grid of points
theta = np.linspace(0, 2 * np.pi, 100)
circle1 = c1 + r1 * np.exp(1j * theta)  # Points on the first circle
circle2 = c2 + r2 * np.exp(1j * theta)  # Points on the second circle


# create something to check orientation
# 20 points of a logarithmic spiral that starts at the origin and spirals outwards
t = np.linspace(0, 4 * np.pi, 20)
my_shape = t * np.exp(1j * t) + 0.3

# append a cloud of points around the beginning of the spiral to 
# signify the "tail" of the "Letter F" shape
tail_points = 0.5 * np.random.rand(100) + 2* np.random.rand(100) * 1j
my_shape = np.concatenate((tail_points, my_shape))

# scale and translate the "Letter F" shape to be insise the first circle
my_shape = my_shape / 50 + c1 + 0.7
# appand a nan to break the line when plotting
my_shape = np.append(np.nan,np.append(my_shape, np.nan))


# for two circles with the same radius, the midline inversion is 
# a reflection about the line that is the perpendicular bisector of the centers of the two circles, so we can just use the reflection function to get the midline inversion in this case

# Use r1 consistently so you don't rely on the 'r' defined in the if-block
if r1 != r2:
    p, q, r, s = c1, c2, r1, r2
    C = p + (r*(q-p))/(r-s)
    # Fixed the bitwise ^ error here too
    R = np.sqrt((r*s*np.abs(q-p)**2)/((r-s)**2) - r*s)
    mid_inv = st.reflection_about(C, R)
else:
    # Ensure this function is actually returning the inner function!
    mid_inv = st.midline_reflection(c1, c2)

basic_inv = st.simple_inversion(c1, r1)
schottky_map = st.compose(mid_inv, basic_inv) 
schottky_map_inv = st.compose(basic_inv, mid_inv) 

tail_points = 1 * np.random.rand(100) - 3 + 5* np.random.rand(100) * 1j
img = tail_points
img1 = mid_inv(tail_points)
img2 = schottky_map(my_shape)
img3 = schottky_map_inv(img2)
#jcircle1 = basic_inv(my_shape)
#img = mid_inv(my_shape)


# plot the original circles and the inverted circle
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.plot(circle1.real, circle1.imag, label='Circle 1 (main)', color='blue')
plt.plot(circle2.real, circle2.imag, label='Circle 2 (2nd Circle)', color='green')
plt.plot(my_shape.real, my_shape.imag, label='fuck this shape', color='black')
plt.plot(img.real, img.imag, label='image of stuff', color='red')
plt.plot(img1.real, img1.imag, label='image of stuff', color='purple')
plt.plot(img2.real, img2.imag, label='image of stuff', color='purple')
plt.plot(img3.real, img3.imag, label='image of stuff', color='pink')
plt.legend()
plt.title('Midline Inversion of Circle 1 about Circle 2')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.axis('equal')
plt.grid()
plt.show()  
