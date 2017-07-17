#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:00:09 2017

@author: yoovrajshinde
"""

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 
import scipy.misc

from navigation import perspect_transform, color_thresh, rover_coords, to_polar_coords

df = pd.read_csv('robot_log.csv', delimiter=';', decimal='.')
csv_img_list = df["Path"].tolist()

# read the ground truth map
ground_truth = mpimg.imread('calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(float)

df["X_Position"].values



### data bucket model to represent the csv data log
class Databucket():
    def __init__(self):
        self.images = csv_img_list
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw  = df["Yaw"].values
        self.count = 0 # current running index
        self.worldmap = np.zeros((200,200,3)).astype(float)
        self.ground_truth = ground_truth_3d
        

# instance of databucket to process the log data
data = Databucket()

def process_image(image):
    # 1) define the source and destination points
    ## src image should be transformed to 10x10 pixel
    dst_size = 5
    ## offset from bottom is required because the image was taken some distance from the camera
    bottom_offset = 10
    img_width = image.shape[1]
    img_height = image.shape[0]
    dst_coordinates = np.float32([[img_width/2 - dst_size, img_height - bottom_offset],
                              [img_width/2 + dst_size, img_height - bottom_offset],
                              [img_width/2 + dst_size, img_height - bottom_offset - 2*dst_size],
                              [img_width/2 - dst_size, img_height - bottom_offset - 2*dst_size]])
    src_coordinates = np.float32([[14,140], [301,140], [200, 96], [118, 96]])
    
    # 2) apply perspective transform
    warped = perspect_transform(image, src_coordinates, dst_coordinates)
    
    # 3) apply color threshold
    # calculating locations which are above threshold of r/g/b
    thresholded_image = color_thresh(warped, rgb_thresh=(160,160,160))
    
    # 4) convert rover centric pixel values to world cordinates
    xpix, ypix = rover_coords(thresholded_image)
    for i in range(len(xpix)):
        data.worldmap[xpix[i].astype(int), ypix[i].astype(int), 1] += 1
    
    output_image = np.zeros((image.shape[0] + data.worldmap.shape[0], image.shape[1]*2, 3))
    
    output_image[0:image.shape[0], 0:image.shape[1]] = image
    
    output_image[0:image.shape[0], image.shape[1]:]  = warped
    
    # adding overlay image of worldmap and groundtruth
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    # flip the image
    output_image[image.shape[0]:, :data.worldmap.shape[1]] = np.flipud(map_add)
    
    # put some text over image
    cv2.putText(output_image, "Populate this image with your analyses to make a video!", (20,20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1
    
    return output_image

'''rover_image = mpimg.imread(data.images[0])
processed_image = process_image(rover_image)
plt.imshow(processed_image)
i = 1
output_file_name = 'output/' + str(i) + '.jpg'
scipy.misc.imsave(output_file_name, processed_image)
'''

for i in range(len(data.images)):
    rover_image = mpimg.imread(data.images[i])
    processed_image = process_image(rover_image)
    output_file_name = 'output/' + str(i) + '.jpg'
    scipy.misc.imsave(output_file_name, processed_image)




##### make the video
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


# Define pathname to save the output video
output = 'output/test_mapping.mp4'
path = 'output/*.jpg'
output_img_list = glob.glob(path)

#data = Databucket() # Re-initialize data in case you're running this cell multiple times
clip = ImageSequenceClip(output_img_list, fps=60) # Note: output video will be sped up because 
                                          # recording rate in simulator is fps=25
#new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
%time clip.write_videofile(output, audio=False)