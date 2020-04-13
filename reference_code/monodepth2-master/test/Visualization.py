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

# Load image -> resize to the size used in our network
im = Image.open("image1.jpg")
img = im.resize((640,192),Image.LANCZOS)
# get np array from image
img = np.array(img.getdata(), dtype= 'uint8').reshape(192,640,3)
#img.dtype

# transform to float -> neccesarry for superpixel algorithms
img = img_as_float(img)

# calculate superpixels and mark the boundaries (same arguments in network)
segments_img = felzenszwalb(img, scale = 120, sigma = 0.8, min_size = 80)
img_bounds = mark_boundaries(img,segments_img)
# transform into uint8 array to transform it into Pillow image
img_bounds = (img_bounds* 255).astype('uint8')

img_bounds = Image.fromarray(img_bounds)
# convert to RGBA to use the image as colormap in surface plot
rgba = img_bounds.convert("RGBA")
rgba = rgba.resize((640,192),Image.LANCZOS)
rgba = np.array(rgba.getdata())
# reshape in the needed format
rgba = rgba.reshape(192,640,4)
#pic.shape

# Normalize the values between 0,1
rgba_norm = (rgba-np.min(rgba))/(np.max(rgba)-np.min(rgba))


# navigate to the folder test 
# load the image and data
path = 'image1_vector_map.npy'
data = np.load(path)
#normals = data.reshape(192,640,3)
xdata = data[0,0,:,:]
ydata = data[0,1,:,:]
zdata = data[0,2,:,:]

#calculate magnitude
mag = np.sqrt(xdata**2 + ydata**2  + zdata**2)

# initialize arrays for the calculated vectors
vec_x = np.zeros_like(xdata)
vec_y = np.zeros_like(ydata)
vec_z = np.zeros_like(zdata)

# array for the properties of superpixels
# std_in_sp = np.empty((3,len(np.unique(segments_img))))
# mean_in_sp = np.empty((3,len(np.unique(segments_img))))
# azi_in_sp = []
# elv_in_sp = []

# vector on window 140,148
# vector on street 71

# loop through all superpixels (SPs)
for i in np.unique(segments_img):
# i = 140
# while i < 141: #np.unique(segments_img).size

    # only take values for the specified SP
    temp_arr = np.where(segments_img == i , mag , 0)

    #calculate the cenetr of mass of the specified SP
    indices = center_of_mass(temp_arr)
    # print(indices)
    # get x and y index
    ix = int(np.round(indices[0]))
    iy = int(np.round(indices[1]))
    # print(ix,iy)
    # set all values outside 1 SP to nan
    temp_xdata = np.where(segments_img == i , xdata , float('nan'))
    temp_ydata = np.where(segments_img == i , ydata , float('nan'))
    temp_zdata = np.where(segments_img == i , zdata , float('nan'))

    # mean_in_sp[:,i] = [np.nanmean(temp_xdata),np.nanmean(temp_ydata),np.nanmean(temp_zdata)]
    # std_in_sp[:,i] = [np.nanstd(temp_xdata),np.nanstd(temp_ydata),np.nanstd(temp_zdata)]

    # calculate the mean value of the x,y,z data and save to vec-variable
    vec_x[ix,iy] = np.nanmean(temp_xdata)
    vec_y[ix,iy] = np.nanmean(temp_ydata)
    vec_z[ix,iy] = np.nanmean(temp_zdata)

    # next lines for calculation of azimuth elviation, mean and std in SPs

    # r = vec_x[ix,iy]**2 + vec_y[ix,iy]**2 + vec_z[ix,iy]**2
    # elv = np.degrees(np.arccos(vec_z[ix,iy]/r))
    # print('elviation %s' % elv)
    # if vec_y[ix,iy] < 0:
    #    azi = 180 - np.degrees(np.arctan(vec_y[ix,iy]/vec_x[ix,iy])) 
    #elif vec_x[ix,iy] < 0 and vec_y[ix,iy] < 0:
    #    azi = np.degrees(np.arctan(vec_y[ix,iy]/vec_x[ix,iy])) - 180
    # else:
    #     azi = np.degrees(np.arctan(vec_x[ix,iy]/vec_y[ix,iy]))
    # print('azimuth %s' % azi)
    # azi_in_sp.append(azi)
    # elv_in_sp.append(elv)
    # i = i +1

#plt.imshow(zdata, cmap='viridis')

# temp_xdata = np.where(segments_img == 148 , xdata , float('nan'))
# temp_ydata = np.where(segments_img == 148 , ydata , float('nan'))
# temp_zdata = np.where(segments_img == 148 , zdata , float('nan'))
# x = np.nanmean(temp_xdata)
# y = np.nanmean(temp_ydata)
# z = np.nanmean(temp_zdata)

# stdx = np.nanstd(temp_xdata)
# stdy = np.nanstd(temp_ydata)
# stdz = np.nanstd(temp_zdata)
# r = temp_xdata**2 + temp_ydata**2 + temp_zdata**2
# stdr = np.nanstd(r)
# meanr = np.nanmean(r)

# elv = np.degrees(np.arccos(temp_zdata/r))
# azi = np.degrees(np.arctan(temp_xdata/temp_ydata))
# meanelv = np.nanmean(elv)
# meanazi = np.nanmean(azi)
# stdelv = np.nanstd(elv)
# stdazi = np.nanstd(azi)

# print('azi = %s +- %s, elv =  %s +- %s, r = %s +- %s, meanx = %s +- %s, meany = %s +- %s , meanz = %s +-%s' %(meanazi, stdazi, meanelv, stdelv, meanr, stdr, x,stdx,y, stdx, z, stdz))

# initialize meshgrid
x = np.arange(640)
y = np.arange(192)
X, Y = np.meshgrid(x, y)

# create 3D figure with specified view
fig = plt.figure()
ax = Axes3D(fig, azim=-103, elev=26)
# ax = Axes3D(fig, azim=-97, elev=27)

# turn off grid
ax.grid(False)

#create array off zeros
zeros = np.zeros_like(zdata)

# plot a XY- plane and used the rgba image as colormap. 
surf = ax.plot_surface(X , Y, zeros , facecolors = rgba_norm , rstride=2, cstride=2, shade=False, alpha = 1)
# surf = ax.plot_surface(X , Y, zeros , facecolors=plt.cm.viridis(mag), rstride=5, cstride=5, shade=False)

# plot the vectors on the imag starting from z=0, scaled arbitrary due to very small values
quiv = ax.quiver(X,Y,zeros, 100*vec_x , 100*vec_y , 100*vec_z, length = 0.4, arrow_length_ratio = 0.4, lw =1, color = 'r')
# quiv = ax.quiver(xdata.flatten(),ydata.flatten(),0,xdata.flatten(),ydata.flatten(),zdata.flatten())

# set axis properties
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0,100)
ax.set_ylim(191,0)
ax.set_xlim(0,640)
ax.set_zticks([])


# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# Plot the surface.
# check: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#wireframe-plots


