#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 18:46:38 2017

@author: yoovrajshinde
"""

import os
os.sys.path
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc
import glob
import imageio
imageio.plugins.ffmpeg.download()

path = 'test_dataset/IMG/*'
img_list = glob.glob(path)

# grab random image 
idx = np.random.randint(0, len(img_list)-1)

# read the image
image = mpimg.imread(img_list[idx])

# plot the image
plt.imshow(image)



###  calibration
example_grid = 'calibration_images/example_grid1.jpg'
example_rock = 'calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(211)
plt.imshow(grid_img)
plt.subplot(212)
plt.imshow(rock_img)

grid_img.shape

## perspective transform

def perspect_transform(img, src, dst):
    
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped


## src image should be transformed to 10x10 pixel
dst_size = 5
## offset from bottom is required because the image was taken some distance from the camera
bottom_offset = 10

img_width = image.shape[1]
img_height = image.shape[0]

src_coordinates = np.float32([[14,140], [301,140], [200, 96], [118, 96]])
dst_coordinates = np.float32([[img_width/2 - dst_size, img_height - bottom_offset],
                              [img_width/2 + dst_size, img_height - bottom_offset],
                              [img_width/2 + dst_size, img_height - bottom_offset - 2*dst_size],
                              [img_width/2 - dst_size, img_height - bottom_offset - 2*dst_size]])


warped = perspect_transform(image, src_coordinates, dst_coordinates)
plt.imshow(warped)
scipy.misc.imsave('output/warped_image.jpg', warped)



#### thresholding 
def color_thresh(img, rgb_thresh=(160,160,160)):
    # preparing the thresholded image
    color_select = np.zeros_like(img[:,:,0])
    
    # calculating locations which are above threshold of r/g/b
    above_thresh = (img[:,:,0] > rgb_thresh[0]) & (img[:,:,1] > rgb_thresh[1]) & (img[:,:,2] > rgb_thresh[2])
    above_thresh
    
    # set the corresponding pixels to white (1)
    color_select[above_thresh] = 1
    return color_select

thresholded_image = color_thresh(warped)
plt.imshow(thresholded_image, cmap='gray')

thresholded_image.nonzero()

#### translation of image
def rover_coords(binary_img):
    xpos, ypos = binary_img.nonzero()
    ## shift the image to rover view and rotate it by 90, so x' = y and y' = x
    y_pixel =  - (xpos - binary_img.shape[1]/2).astype(np.int)
    x_pixel =  - (ypos - binary_img.shape[0]).astype(np.int)
    return x_pixel, y_pixel

xpix, ypix = rover_coords(thresholded_image)

#### convert cartesian coordinates to polar coordinates
def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

dist, angles = to_polar_coords(xpix, ypix)

#### overall angle by which the rover should move
mean_dir = np.mean(angles)

### plotting all the figures
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.imshow(image)

plt.subplot(222)
plt.imshow(warped)

plt.subplot(223)
plt.imshow(thresholded_image, cmap='gray')

plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)


target_pixels = np.column_stack((xpix,ypix, np.zeros([len(xpix)])))
target_pixels = tuple(map(tuple,target_pixels))


color_select = np.zeros_like(image[:,:,0])
rgb_thresh=(160,160,160)
# calculating locations which are above threshold of r/g/b
above_thresh = (image[:,:,0] > rgb_thresh[0]) & (image[:,:,1] > rgb_thresh[1]) & (image[:,:,2] > rgb_thresh[2])
above_thresh
