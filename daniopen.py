import cv2
import numpy as np
from numpy import genfromtxt
import glob
import os
import math
import matplotlib.pyplot as plt

#### preferences
img_extension = ".png"

data_dir = "/home/user/data/public_test/"
results_dir = "/home/user/results/public_test/"

dirstring = data_dir + "/" + "*" + img_extension;
img_list =glob.glob(dirstring); img_list.sort(key=os.path.getmtime); print(img_list)
# set start and end frame; default is the entire directory contents (0 - len(img_list))
minframe = 0; maxframe = len(img_list)

threshold = 120 #threshold for binarizing image
min_size_thresh = 50 #minimum pixel area for fish size

#image processing settings
crop_frame = False; resize_frame = True; resize_factor = 3

#### set boundaries if cropping image
x_min = 0; x_max = 270
y_min = 0; y_max = 726

### duration of frame display; 0 will pause between frames, 1 is max speed
delay = 1

#### data handling functions
def create_path(path):
    if (os.path.exists(path) == False):
        os.makedirs(path); path_created = True
    else: path_created = False
    return path_created

def write_text(handle, text_path, data_point, data_type):
    handle = file(text_path, 'a')
    np.savetxt(handle, data_point, delimiter=',', fmt=data_type)
    handle.close()

### image processing functions
def process_frame(frame):
    if resize_frame == True:
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    if crop_frame == True:
        frame = frame[y_min:y_max, x_min:x_max]
    return frame

def center_of_mass(frame, com):
    # Convert to grayscale and threshold
    raw_frame = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    # Detect contour in image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) > 0):
        contoursize = [np.nan] * len(contours)
    # If multiple contours were detected, then select the largest one
        for contourNo in range(0, len(contours)):
            contoursize[contourNo] = cv2.contourArea(contours[contourNo])
    # If contour size is greater than the size threshold, calculate its center
    # of mass.
        if (np.max(contoursize) > min_size_thresh):
            max_cont = np.max(contoursize)
            max_index = (np.where(contoursize == np.max(contoursize))[0])
            if (len(max_index) > 1):
                max_index = 0
            M = cv2.moments(contours[max_index])
            if (M['m10'] > 0) and (M['m00'] > 0) and (M['m01'] > 0):
                com = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    # Draw circles on frame for bugfixing
                height,width,depth = frame.shape
                frame = np.zeros((height,width,3), np.uint8)
                frame = 255-frame
                cv2.drawContours(frame, contours, max_index, (69,96,22), -1)
                cv2.circle(frame, com, 5, (0, 255, 255), -1)
        else: max_index = 0
    elif (len(contours) == 0): contours = [0]; max_index = 0
    return (frame, thresh, com, raw_frame, contours[max_index])

def body_length(cnt):
    extrema=[(cnt[cnt[:,:,0].argmin()][0]),(cnt[cnt[:,:,0].argmax()][0]),(cnt[cnt[:,:,1].argmin()][0]),(cnt[cnt[:,:,1].argmax()][0])]
    all_dist = [np.linalg.norm(extrema[0]-extrema[1]), np.linalg.norm(extrema[0]-extrema[2]), np.linalg.norm(extrema[0]-extrema[3]), np.linalg.norm(extrema[1]-extrema[2]),np.linalg.norm(extrema[1]-extrema[3]),np.linalg.norm(extrema[2]-extrema[3])]
    axis_length = np.max(all_dist)
    return extrema, all_dist, axis_length

def orientation_detect(img,cnt,com,extrema,all_dist):
    max_axis = np.where(np.max(all_dist) == all_dist)[0][0]
    #dictionary of all possible comparisons based on index of maximum axis length
    index_dict={0:[0,1],1:[0,2],2:[0,3],3:[1,2],4:[1,3],5:[2,3]}
    for index in range (0,len(index_dict)):
        first = index_dict[index][0]; second = index_dict[index][1]
        if (np.linalg.norm(extrema[first] - com) < np.linalg.norm(extrema[second] - com)):
            max_index = first;
        else: max_index = second
    angle = (math.atan2(extrema[max_index][1] - com[1], com[0] - extrema[max_index][0]))*(180/3.141592)+180
    cv2.line(img,com,tuple(extrema[max_index]),(100,100,255),2)
    cv2.circle(img,com,2,(255,255,255),-1)
    for index in range(0,len(extrema)):
        cv2.circle(img,tuple(extrema[index]),1,(0,0,255),-1)
	#cv2.putText(img,str(int(angle)),(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255))
    return img, angle

### plotting functions
def xy_plot(frame,input_file,output_file):
# Generate graph based on dimensions of input frame
    height,width,depth = frame.shape
    output_height = height; output_width = width
    graph = np.zeros((output_height,output_width,3), np.uint8); graph = 255-graph
    resize_x = 1; resize_y = 1

### RED TRACE
    #start_color = [128,164,255]; end_color = [0,52,184]
### GREEN TRACE
    start_color = [196,230,138]; end_color = [69,96,22]
### BLUE TRACE
    #start_color = [229,218,214]; end_color = [184,81,41]
    color = start_color
# Generate filename of xy plot
    content = genfromtxt(input_file, delimiter=',')

    line_thickness = 1; circle_size = 1
    start_slice = 0; end_slice = len(content); slice_number = abs(end_slice - start_slice)
    increment = [float(abs(start_color[0] - end_color[0])/float(slice_number)),float(abs(start_color[1] - end_color[1])/float(slice_number)),float(abs(start_color[1] - end_color[1])/float(slice_number))]

# Create x, y and distance arrays based on length of file
    x = [np.nan] * (slice_number)
    y = [np.nan] * (slice_number)
    distance = [np.nan] * (slice_number)

    for i in range(0,slice_number):
        x[i] = int((content[i+start_slice][0])) * resize_x
        y[i] = int((content[i+start_slice][1])) * resize_y

        if i > 0:
            if x[i] < 0: x[i] = prev_x
            if y[i] < 0: y[i] = prev_y
        point = np.array([x[i], y[i]])

    # Draw center of mass
        cv2.circle(graph, (x[i], y[i]), circle_size, color, -1)
    # Draw lines between points as long as we've drawn at least one point already
        if i > 0:
            cv2.line(graph, (prev_x, prev_y), (x[i], y[i]), color, thickness=line_thickness)
            distance[i - 1] = np.linalg.norm(prev_point - point)
            #shift colors
            color[0] = float(color[0] - increment[0])
            color[1] = float(color[1] - increment[1])
            color[2] = float(color[2] - increment[2])
    # Increment points for line drawing in next frame
        prev_x = x[i]; prev_y = y[i]; prev_point = np.array([x[i], y[i]])
    #output plot
    cv2.imwrite(output_file, graph)

def polar_plot(input_file, output_file):
        calzone_plot = False
        print("Generating polar plots")
        bins = 360; bin_size = 360/bins
        bottom = 0; radii = 0
        angle_counts = [0] * bins
        theta = np.linspace(0.0, 2 * np.pi, bins, endpoint=False)
        data = genfromtxt(input_file,delimiter=',')

#Collapse angles into calzone plot if True
        if calzone_plot == True:
            for i in range(0,len(data)):
                if data[i][0] > 180:
                    data[i][0] = (data[i][0]-180)%360
                    data[i][0] = np.abs(data[i][0] - 180)
        prev_angle = 0; angle = 0 + bin_size
#Histogram
        for bin_no in range(0,bins):
            for row in range(0,len(data)):
                if np.isnan((data[row][0])) == False:
                    if (int(data[row][0]) < angle) and (int(data[row][0]) >= prev_angle):
                        angle_counts[bin_no] = angle_counts[bin_no] + 1
            prev_angle = angle; angle = angle + bin_size
#Plot settings
        max_height = np.max(angle_counts)
        radii = angle_counts
        width = (2*np.pi) / bins
        ax = plt.subplot(111, polar=True)
        ax.yaxis.set_visible(False)
        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        ax.set_rmax(max_height)

    # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.Greens(r))
            bar.set_alpha(0.8)
        plt.savefig(output_file); plt.clf()


#### MAIN SCRIPT START
# initialize variables
com = [-1,-1,-1]
angle_part = [np.nan] * 2
kernel = np.ones((5,5), np.uint8)

# Create directory to save results
create_path(results_dir)

for frame_number in range(minframe,maxframe):
    section_occupancy = -1;
    filename = data_dir + str(frame_number).zfill(4) + ".png"
    frame = cv2.imread(filename)
    frame = process_frame(frame)
    height,width,depth=frame.shape

    if frame_number > 1:
        prev_com = com
        frame, thresh, com, raw_frame, contour = center_of_mass(frame, com)
        write_text("part_handle", results_dir + "_xy.txt", [com], "%d")

        if (np.any(contour) == True) and (frame_number > minframe) and (com[0] > -1) and (prev_com[0] > -1):
            extrema,all_dist,axis_length=body_length(contour)
            orientation,angle = orientation_detect(frame,contour,com,extrema,all_dist)
            angle_part[0] = angle; angle_part[1] = int(frame_number)
            cv2.imshow('Frame',orientation)

        else: angle_part[0] = -1; angle_part[1] = int(frame_number)
        write_text("angle_handle", results_dir + "_angle.txt", [angle_part], "%d")


# Use 'q' to terminate capture (REQUIRED FOR IMAGE DISPLAY)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

xy_plot(frame,results_dir + "_xy.txt", results_dir + "_xy.png")
polar_plot(results_dir + "_angle.txt", results_dir + "_polarplot.png")
