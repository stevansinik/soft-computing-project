#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:37:18 2019

@author: Stevan Sinik
"""

from skimage import morphology as morph
import numpy as np

# Video processing properties

line_width = 9

binarization_threshold = 25

binarization_upper_value = 255

digits_and_noise_radius = 30.0

noise_radius = 5.0

hough_rho_precision = 1

hough_phi_precision = np.pi/180

hough_threshold = 250

hough_minimum_line_length = 50

hough_maximum_line_gap = 5

kernel = morph.disk(1.5)

# Digit recognition neural network properties

mnist_digit_edge = 28

mnist_digit_area = mnist_digit_edge * mnist_digit_edge

number_of_hidden_neurons = 512

dropout_rate = 0.2

number_of_classes = 10

optimizer='adam'

loss='categorical_crossentropy'

metrics=['accuracy']

batch_size=128

epochs=20

validation_split=.1
