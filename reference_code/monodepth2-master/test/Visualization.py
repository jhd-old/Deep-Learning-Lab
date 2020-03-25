# 3D Visualization of normal vectors
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
import numpy as np 
from scipy.ndimage.measurements import center_of_mass

im = Image.open("image1.jpg")
img = im.resize((640,192),Image.LANCZOS)
img = np.array(img.getdata(), dtype= 'uint8').reshape(192,640,3)
#img.dtype

img = img_as_float(img)
segments_img = felzenszwalb(img, scale = 100, sigma = 0.5, min_size = 50)
img_bounds = mark_boundaries(img,segments_img)
img_bounds = (img_bounds* 255).astype('uint8')

img_bounds = Image.fromarray(img_bounds)
rgba = img_bounds.convert("RGBA")
rgba = rgba.resize((640,192),Image.LANCZOS)
rgba = np.array(rgba.getdata())
rgba = rgba.reshape(192,640,4)
#pic.shape

# Normalize the values between 0,1
rgba_norm = (rgba-np.min(rgba))/(np.max(rgba)-np.min(rgba))


#navigate to folder before
path = 'image1_vector_map.npy'
data = np.load(path)
#normals = data.reshape(192,640,3)
xdata = data[0,0,:,:]
ydata = data[0,1,:,:]
zdata = data[0,2,:,:]

mag = np.sqrt(xdata**2 + ydata**2  + zdata**2)
#elviation = np.arccos(zdata/mag)
#azimuth = np.arctan(ydata/xdata)
# for x < 0 -> + pi, for x < 0 & y < 0 -> - pi
#np.round(10*mag[0,0]*np.sin(elviation[0,0])*np.cos(azimuth[0,0]))
#np.round(10*mag[0,0]*np.sin(elviation[0,0])*np.sin(azimuth[0,0]))

vec_x = np.zeros_like(xdata)
vec_y = np.zeros_like(ydata)
vec_z = np.zeros_like(zdata)

for i in np.unique(segments_img):
    temp_arr = np.where(segments_img == i , mag , 0)
    indices = center_of_mass(temp_arr)
    #print(indices)
    ix = int(np.round(indices[0]))
    iy = int(np.round(indices[1]))
    #print(ix,iy)
    temp_xdata = np.where(segments_img == i , xdata , float('nan'))
    temp_ydata = np.where(segments_img == i , ydata , float('nan'))
    temp_zdata = np.where(segments_img == i , zdata , float('nan'))
    vec_x[ix,iy] = np.nanmean(temp_xdata)
    vec_y[ix,iy] = np.nanmean(temp_ydata)
    vec_z[ix,iy] = np.nanmean(temp_zdata)
    #r = vec_y[ix,iy]**2 + vec_y[ix,iy]**2 + vec_z[ix,iy]**2
    #elv = np.arccos(vec_z[ix,iy]/r)
    #print(np.degrees(elv))
    
#plt.imshow(zdata, cmap='viridis')

x = np.arange(640)
y = np.arange(192)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = Axes3D(fig)

zeros = np.zeros_like(zdata)

surf = ax.plot_surface(X , Y, zeros , facecolors = rgba_norm , rstride=2, cstride=2, shade=False)
#surf = ax.plot_surface(X , Y, zeros , facecolors=plt.cm.viridis(mag), rstride=5, cstride=5, shade=False)
#quiv = ax.quiver(xdata.flatten(),ydata.flatten(),0,xdata.flatten(),ydata.flatten(),zdata.flatten())
quiv = ax.quiver(X,Y,zeros, vec_x , vec_y , vec_z, arrow_length_ratio = 0.3, lw =1, color = 'r')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0,10)
ax.set_ylim(191,0)
ax.set_xlim(0,640)



# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# ytest = ydata/np.max(ydata)
# xtest = xdata/np.max(xdata)
# ztest = zdata/np.max(zdata)

# d = 0
# Z = ( - xdata*X - ydata*Y) / zdata

# mean = np.mean(Z)
# std = np.std(Z)

# for i in range(Z.shape[0]):
#     for j in range(Z.shape[1]):
#         if Z[i,j] > mean+2*std or Z[i,j] < mean -2*std:
#             Z[i,j] = 0 

# Plot the surface.
# check: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#wireframe-plots


