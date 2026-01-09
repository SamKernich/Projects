""" 
This code is used to animate the position of the particles in the particle accelerator, based on simulated readings
from beam position monitors that are spread in intervals around the accelerator. 

Created: 12/05/2025 by Sam Kernich
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define ring parameters
R = 5  # Radius of the ring
num_frames = 100  # Number of frames for animation

# Create theta values for animation
theta_vals = np.linspace(0, 2*np.pi, num_frames)

# Initialize figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
ax.set_xlim([-R-1, R+1])
ax.set_ylim([-R-1, R+1])
ax.set_zlim([-1, 1])  # Keep z-axis centered

# Plot the ring (static)
t = np.linspace(0, 2*np.pi, 100)
ax.plot(R * np.cos(t), R * np.sin(t), np.zeros_like(t), 'gray', linewidth=1.5)

# Initialize the particle
particle, = ax.plot([], [], [], 'ro', markersize=8)  # Red dot

# Update function for animation
def update(frame):
    x = R * np.cos(theta_vals[frame]) 
    y = R * np.sin(theta_vals[frame]) 
    z = 0  # Fixed in the XY plane
    particle.set_data(x, y)
    particle.set_3d_properties(z)
    return particle,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

# Show the animation
plt.show()
