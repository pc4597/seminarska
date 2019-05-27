import os
import numpy
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Load the STL files
your_mesh = mesh.Mesh.from_file('ponvica.stl')

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Auto scale to the mesh size
scale = your_mesh.points.flatten('F')
axes.auto_scale_xyz(scale, scale, scale)

# Evaluating Mesh properties (Volume, Center of gravity, Inertia)
volume, cog, inertia = your_mesh.get_mass_properties()
print("Volume                                  = {0}".format(volume))
print("Position of the center of gravity (COG) = {0}".format(cog))
print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
print("                                          {0}".format(inertia[1,:]))
print("                                          {0}".format(inertia[2,:]))

# Read points forming the model
pts = your_mesh.points
print('This models consists of {} points'.format(pts.shape[0]))

# Get surface normals at each point
your_mesh.normals
print(your_mesh.normals)

"""tockeABP = []
def onclick(event):
    if event.key == 'shift': 
        x, y, z = event.xdata, event.ydata, event.zdata
        tockeABP.append((x, y, z))
        ax.plot(x, y, z, 'or')
        fig.canvas.draw()
        
        
fig = pyplot.figure()
ax = fig.add_subplot(111)
ax.imshow(pts)    
ka = fig.canvas.mpl_connect('button_press_event', onclick)"""

# Show the plot to the screen
pyplot.show()
