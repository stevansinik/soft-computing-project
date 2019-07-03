#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:23:12 2019

@author: Stevan Sinik
"""

import cv2
import numpy as np

from skimage import morphology as morph
from matplotlib import pyplot as plt
from keras.models import model_from_json    
    
# Code snippets
    
# OpenCV image combinations; values don't go out of bounds

    # Add
    a_plus_b = cv2.add(a, b)
    
    # Subtract
    a_minus_b = cv2.subtract(a, b)

# OpenCV get monochromatic image
gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

# OpenCV global threshold
threshold_value, binary = cv2.threshold(gray, threshold_factor, value_to_convert_to, cv2.THRESH_BINARY)
    # e.g. threshold_value, binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
# OpenCV contour retrieval
contours_image_modified, contours, hierarchy = cv2.findContours(contours_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# OpenCV contour properties
x,y,w,h = cv2.boundingRect(contour)
area = cv2.contourArea(contour)

# OpenCV drawing

    # Line
    cv2.line(image,(x1,y1),(x2,y2),line_color,line_thickness)
        # e.g. cv2.line(bgr_image,(x1,y1),(x2,y2),(0,255,0),2)
   
    # Rectangle     
    cv2.rectangle(image, (x1, y1), (x2, y2), rectangle_color, rectangle_thickness)
        # e.g. cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
# OpenCV Hough line (rho, theta) to OpenCV line ((x1, y1), (x2, y2))
rho, theta = hough_line
a = np.cos(theta)
b = np.sin(theta)
x0 = a*rho
y0 = b*rho
x1 = int(x0 + 1000*(-b))
y1 = int(y0 + 1000*(a))
x2 = int(x0 - 1000*(-b))
y2 = int(y0 - 1000*(a))

# OpenCV initialize video input
video_in = cv2.VideoCapture(video_path)

# OpenCV index frame to capture/decode next
video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    # for more info on VideoCapture class properties see:
    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d

# OpenCV initialize video output
video_out = cv2.VideoWriter('<file_name>.avi', cv2.VideoWriter_fourcc(four_CC), fps, video_format)
    # e.g. out = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'XVID'), 40.0, (640, 480))

# Numpy array of x-es
array_of_x = np.ones(shape, np.uint8) * x
    # e.g. x = 40, shape = (2, ) array_of_x = [40, 40]

# Numpy logical and
a_and_b = np.logical_and(a, b)
    
# Numpy boolean to 0 and 255 gray
gray = binary.astype(np.uint8)
gray *= 255

# Numpy gray to 3 channel; necessary for video output
image_3_channel = np.stack((gray,)*3, axis=-1)

# Numpy split channels
red = rgb[:,:,0]
green = rgb[:,:,1]
blue = rgb[:,:,2]

# Numpy write array to .txt file
np.savetxt("<filename>.txt", numpy_array, format_string)
    # e.g. np.savetxt("array.txt", numpy_array, fmt="%d")

# Morphology 

    # Dilation
    dilated_binary = morph.binary_dilation(binary, structuring_element)

    # Opening
    opened_binary = morph.binary_opening(binary, structuring_element)

    # Closing
    closed_binary = morph.binary_closing(binary, structuring_element)
    
# Pyplot set plot title
plt.title(title)
