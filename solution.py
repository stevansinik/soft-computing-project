#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 23:06:36 2019

@author: Stevan Sinik
"""

import os
import datetime

import cv2
import numpy as np

from skimage import morphology as morph
from matplotlib import pyplot as plt
from keras.models import model_from_json

import parameters
    
def get_avg_line(lines):
    rho_sum = 0
    theta_sum = 0
    count = 0
    for line in lines:
        for rho, theta in line:
            rho_sum += rho
            theta_sum += theta
            count += 1
    
    return (rho_sum/count, theta_sum/count)
    
def get_video_paths(video_directory):
    video_files = [name for name in os.listdir(video_directory)]
    video_files = sorted(video_files, key=lambda name: int(name[name.index("-") + 1 : name.index(".")]))
    video_paths = [os.path.join(video_directory, video_file) for video_file in video_files]
    return video_paths
        
def load_neural_network(json_file_path, weights_file_path):
    global neural_network
    json_file = open(json_file_path, 'r')
    json = json_file.read()
    json_file.close()
    neural_network = model_from_json(json)
    neural_network.load_weights(weights_file_path)
    neural_network.compile(parameters.optimizer, parameters.loss, parameters.metrics)
    
def display_frame(frame, frame_type):
    plt.figure()
    if frame_type == "RGB":
        plt.imshow(frame)        
    elif frame_type == "BGR":
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    elif frame_type == "gray":
        plt.imshow(frame, "gray")
    elif frame_type == "binary":
        shown_frame = binary_to_monochrome(frame)
        plt.imshow(shown_frame, "gray")
            
    
def read_input_capture(input_capture, max_frames_count, stride):
    global max_iterations
    
    frames = []
    
    iteration_count = 0
    ret = True
    while ret:
        if max_iterations != None and iteration_count >= max_iterations:
            return
                    
        for i in range(max_frames_count):
            ret, frame = input_capture.read()
            if not ret:
                break
            frames.append(frame)
            
        if len(frames) > 0:
            yield frames
            
        del frames[0:min(len(frames), stride)]
        
        iteration_count += 1
        
def erase_line(frame, line_coordinates, line_width):
    black = (0, 0, 0)
    cv2.line(frame, 
                 (line_coordinates[0], line_coordinates[1]), 
                 (line_coordinates[2], line_coordinates[3]),
                 black,
                 line_width)
    
def frame_transformations(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_parameter, binary_frame = cv2.threshold(
            gray_frame, 
            parameters.binarization_threshold, 
            parameters.binarization_upper_value, 
            cv2.THRESH_BINARY)
    return (gray_frame, binary_frame)

def binary_to_monochrome(binary_frame):
    monochrome_frame = binary_frame.astype(np.uint8)
    monochrome_frame *= 255
    return monochrome_frame

def monochrome_to_video(monochrome_frame):
    video_frame = np.stack((monochrome_frame,)*3, axis=-1)    
    return video_frame

def binary_to_video(binary_frame):
    monochrome_frame = binary_to_monochrome(binary_frame)
    return monochrome_to_video(monochrome_frame)  

def predict_digit(image_fragment):
    global neural_network
    fragment_shape = np.array(image_fragment.shape, np.int8)
    offsets = (parameters.mnist_digit_edge - fragment_shape) // 2
    input_matrix = np.zeros((parameters.mnist_digit_edge, parameters.mnist_digit_edge), np.float32)
    input_matrix[offsets[0]:offsets[0]+fragment_shape[0], offsets[1]:offsets[1]+fragment_shape[1]] = image_fragment
    input_matrix /= 255.0
    input_matrix = input_matrix.reshape(1, parameters.mnist_digit_area)
    output = neural_network.predict(input_matrix)
    output = np.squeeze(np.asarray(output))
    max_index = np.argmax(output)
    return (max_index, output[max_index])

def point_inside(point, rectangle):
    x = point[0]
    y = point[1]
    rx1 = rectangle[0]
    rx2 = rx1 + rectangle[2]
    ry1 = rectangle[1]
    ry2 = ry1 + rectangle[3]
    return x >= rx1 and x <= rx2 and y >= ry1 and y <= ry2

def filter_contours_by_radius(contours, radius):
    return [contour for contour in contours if cv2.minEnclosingCircle(contour)[1] >= radius] 

def cannonic_line_form(line_coordinates):
    p1 = None
    p2 = None
    if(line_coordinates[0] <= line_coordinates[2]):
        p1 = line_coordinates[0:2]
        p2 = line_coordinates[2:4]
    else:
        p1 = line_coordinates[2:4]
        p2 = line_coordinates[0:2]
    k = (p1[1] - p2[1]) / (p1[0] - p2[0])
    n = p1[1] - k * p1[0]
    return (k, n)

def is_above_line(x, y, k, n):
    return y < (k * x + n)

def get_colliding_zones(retained_contours, line_rect, line_k, line_n):
    colliding_zones = []
    for contour in retained_contours:
        x, y, w, h = cv2.boundingRect(contour)
        mid_x = x + w / 2
        mid_y = y + h / 2
        center = (mid_x, mid_y)
        size = (w, h)
        rotation = 0
        rotated_form = (center, size, rotation)
        
        if (cv2.rotatedRectangleIntersection(line_rect, rotated_form)[0] == 1) and (is_above_line(mid_x, mid_y, line_k, line_n)):
            colliding_zones.append(((x, y, w, h), center))
            
    return colliding_zones
    
def supress_duplicate_detections(new_zones, old_zones):
    previous_frame_zones = list(old_zones)
    this_frame_zones = list(new_zones)
    
    old_zones.clear()
    new_zones.clear()
    
    for this_frame_zone in this_frame_zones:
        old_zones.append(this_frame_zone)
        
        this_frame_zone_center = this_frame_zone[1]        
        is_new_zone = True

        for i in range(len(previous_frame_zones)):
            previous_frame_zone_rectangle = previous_frame_zones[i][0]
            if point_inside(this_frame_zone_center, previous_frame_zone_rectangle):
                del previous_frame_zones[i]                
                is_new_zone = False
                break
        
        if is_new_zone:
            new_zones.append(this_frame_zone)

def detect_line(initial_frame, index):
    
    initial_gray, initial_binary = frame_transformations(initial_frame)
    
    initial_binary = morph.dilation(initial_binary, parameters.kernel)
        
    _, initial_contours, _ = cv2.findContours(initial_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    line_mask = np.zeros(initial_binary.shape, np.uint8)
    
    retained_contours = filter_contours_by_radius(initial_contours, parameters.digits_and_noise_radius)
    
    for contour in retained_contours:
        cv2.drawContours(line_mask, [contour], 0, (1), -1)
        
    lines = cv2.HoughLinesP(
            line_mask, 
            parameters.hough_rho_precision, 
            parameters.hough_phi_precision, 
            parameters.hough_threshold, 
            None,
            parameters.hough_minimum_line_length,
            parameters.hough_maximum_line_gap)
    
    lines_shape = lines.shape
    rows = lines_shape[0]
    cols = lines_shape[2]
    lines = np.reshape(lines, (rows, cols))
    lines = np.round(np.sum(lines, axis=0) / rows).astype(np.int)
    
    line_coordinates = np.reshape(lines, 4)
    
    cv2.line(initial_frame, 
                 (line_coordinates[0], line_coordinates[1]), 
                 (line_coordinates[2], line_coordinates[3]),
                 (0, 0, 255),
                 parameters.line_width)

    pushed_cwd = os.getcwd()
    os.chdir(os.path.dirname(pushed_cwd))
    cv2.imwrite('result_{}.png'.format(index), initial_frame)
    os.chdir(pushed_cwd)
    
    return line_coordinates

def solve_video(video_index, video_path, output_directory):
    os.chdir(output_directory)
    video_directory = os.path.join(output_directory, "Video_" + str(video_index))
    os.mkdir(video_directory)
    os.chdir(video_directory)

    input_capture = cv2.VideoCapture(video_path)
    visualizing_writer = cv2.VideoWriter('tracking.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
    
    _, initial_frame = input_capture.read()
    
    line_coordinates = detect_line(initial_frame, video_index)
    
    frame_shape = initial_frame.shape
    line_mask = np.ones((frame_shape[0:2]), dtype=np.uint8)
    cv2.line(line_mask, 
                 (line_coordinates[0], line_coordinates[1]), 
                 (line_coordinates[2], line_coordinates[3]),
                 (0),
                 parameters.line_width)
    inv_line_mask = 1 - line_mask
    
    _, contours, _ = cv2.findContours(inv_line_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minRect = cv2.minAreaRect(contours[0])
    
    line_k, line_n = cannonic_line_form(line_coordinates)
    
    total = 0
    
    old_zones = []
    new_zones = []
    
    detection_counter = 0
    
    for frames_list in read_input_capture(input_capture, 1, 1):
        frame = frames_list[0]
        erase_line(frame, line_coordinates, parameters.line_width)
        gray, binary = frame_transformations(frame)
        
        binary = morph.dilation(binary, parameters.kernel)
        
        _, contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(binary.shape, np.uint8)
        
        retained_contours = filter_contours_by_radius(contours, parameters.noise_radius)
        
        for contour in retained_contours:
            cv2.drawContours(mask, [contour], 0, (1), -1)

        #for i in range(3):
            #frame[:,:,i] *= mask
            
        gray *= mask
                     
        
        new_zones = get_colliding_zones(retained_contours, minRect, line_k, line_n)
        supress_duplicate_detections(new_zones, old_zones)
        
        digits = []
        for new_zone in new_zones:
            x, y, w, h = new_zone[0]
            fragment = gray[y:y+h, x:x+w]
            digit = predict_digit(fragment)[0]
            total += digit
            digits.append(digit)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            
            
        detected_string = 'Detected: '
        for digit in digits:
            detected_string += " " + str(digit)
        
        cv2.putText(frame, detected_string, (0,440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)            
            
        
        cv2.putText(frame, 'Total: {}'.format(total), (0,470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)            

        if len(digits) > 0:
            cv2.imwrite('detection_{}.png'.format(detection_counter), frame)
            detection_counter += 1
        
        visualizing_writer.write(frame)
    
    input_capture.release()
    visualizing_writer.release()    
    
    return total
    
    
def write_output(file_path, sums):
    lines = ["Stevan Sinik",
             "file	sum"]
    lines.extend('video-{}.avi\t{}'.format(index, one_sum) for index, one_sum in enumerate(sums))
    
    with open(file_path, 'w+') as file:
        for line in lines:
            file.write(line + '\r\n')
        file.flush()

neural_network = None
max_iterations = None
    
if __name__ == "__main__":
    
    solution_directory = None
    video_root_directory = None
    output_root_directory = None
    neural_network_structure_path = None
    neural_network_weights_path = None
    
    solution_directory = os.getcwd()
    
    if video_root_directory is None:
        video_root_directory = os.path.join(solution_directory, 'input_videos')
        
    if output_root_directory is None:
        output_root_directory = os.path.join(solution_directory, 'output')
        
    if neural_network_structure_path is None:
        neural_network_structure_path = os.path.join(solution_directory, 'ann_structure.json')
        
    if neural_network_weights_path is None:
        neural_network_weights_path = os.path.join(solution_directory, 'ann_weights.h5')
        
    now = datetime.datetime.now()    
    output_directory = os.path.join(output_root_directory, "Run-" + now.strftime("%Y-%m-%d-T-%H-%M-%S"))
    
    os.chdir(output_root_directory)
    os.mkdir(output_directory)

    load_neural_network(neural_network_structure_path, neural_network_weights_path)
    
    video_paths = get_video_paths(video_root_directory)
    sums = ["Not processed"] * len(video_paths)
    
    enumerated_video_paths = [index_and_path for index_and_path in enumerate(video_paths)]
    for index, video_path in enumerated_video_paths[:]:
        print('Processing video {} ...'.format(index))
        try:
            output_sum = solve_video(index, video_path, output_directory)
            sums[index] = output_sum
        except Exception as e:
            sums[index] = "Exception raised"
        
            print(e)
        
    write_output(os.path.join(output_directory, 'out.txt'), sums)
