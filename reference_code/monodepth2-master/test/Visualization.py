# 3D Visualization of normal vectors
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3Dimport matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

#navigate to folder before
path = 'image3_vector_map.npy'
data = np.load(path)
normals = data.reshape(192,640,3)
xdata = data[:,:,0]
ydata = data[:,:,1]
zdata = data[:,:,2]

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d')

x = np.linspace(0, 1, 640)
y = np.linspace(0, 1, 192)
X, Y = np.meshgrid(x, y)

# Plot the surface.
# check: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#wireframe-plots

surf = ax.plot_surface(xdata , ydata, zdata , cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# surf =  ax.contour3D(xdata, ydata, zdata, 50, cmap='binary')
# surf =  ax.plot_surface(xdata, ydata, zdata, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

