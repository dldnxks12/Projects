import os   # Terminal Control
import sys  # Terminal Control 

import cv2                     # QR Code 
import pyzbar.pyzbar as pyzbar # QR Code

import math
import time
import pickle
import logging    # For Logger
import subprocess # For parallel Processing 
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs

# Get Robot Position (Area = 1/2 * width * height)
def cal_dist(x1, y1, x2, y2, centerX, centerY): 
	Triangle_Area = abs( (x1-centerX)*(y2-centerY) - (y1-centerY)*(x2-centerX) )
	line_distance = math.dist((x1,y1), (x2, y2))
	distance = (2*Triangle_Area) / line_distance 
	return distance

# Get Angle
def get_angle(Points):
	angle = (np.arctan2(Points[1] - Points[3], Points[0] - Points[2]) * 180) / np.pi 	
	return angle

# Get ROI
def ROI(img, vertices, color3 = (255, 255, 255), color1 = 255):
	mask = np.zeros_like(img)
	if len(img.shape) > 2: # 3 channel image
		color = color3
	else:
		color = color1        	
	cv2.fillPoly(mask, vertices, color)
	ROI_IMG = cv2.bitwise_and(img, mask)
	return ROI_IMG

# Get Main line
def get_fitline(img, f_lines): # 대표선 구하기   
    lines = np.squeeze(f_lines, 1)
    lines = lines.reshape(lines.shape[0]*2,2)
    rows,cols = img.shape[:2]
    output = cv2.fitLine(lines,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    
    result = [x1,y1,x2,y2]
    return result

# GPU Setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('log.txt', mode = 'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ----  Global Vars ---- #
# Temps
L_x1 = [] # Points temp
L_y1 = [] # Points temp
L_x2 = [] # Points temp
L_y2 = [] # Points temp
R_x1 = [] # Points temp
R_y1 = [] # Points temp
R_x2 = [] # Points temp
R_y2 = [] # Points temp
Left_Avg_points_temp  = 0
Right_Avg_points_temp = 0

# Angle
Base_angle       = 90
Left_Base_angle  = 110 
Right_Base_angle = 80
Left_Angle       = 90
Right_Angle      = 90
Reference_angle  = 0 

# Position
Base_left_distance  = 400  # Init Left Position
Base_right_distance = 400  # Init Right Position 
Left_distance       = 0    # Current Pos 
Right_distance      = 0    # Current Pos 

# Control
Left_direction   = 1
Right_direction  = 1
Left_Difference  = 0 
Right_Difference = 0
Base_weight      = 100
Left_Wheel       = 0
Right_Wheel      = 0
Send_Left_Wheel  = ""
Send_Right_Wheel = ""

# Flags
Camera_init               = False
Init_distance             = True
Left_line_interpolation   = 0
Right_line_interpolation  = 0
No_line_flag              = 0
No_left_line_flag         = 0
No_right_line_flag        = 0
Left_Pos_flag             = 0
Left_Angle_flag           = 0 
Right_Pos_flag            = 0  
Right_Angle_flag          = 0 


# Moving Average
Left_pos_temp     = []    # List for Position Moving Average
Left_angle_temp   = []    # List for Angle Moving Average
Right_pos_temp    = []    # List for Position Moving Average
Right_angle_temp  = []    # List for Angle Moving Average
Left_Avg_Pos      = 0     # Average Position
Left_Avg_Ang      = 0     # Average Angle
Right_Avg_Pos     = 0     # Average Position
Right_Avg_Ang     = 0     # Average Angle

# Middle Points 
STM_Weight_Value = 1.5
STM_Ratio        = 0
Road_line        = 0
Zero_gap         = 0 
Left_check       = 0
Right_check      = 0


# Flag 
Stop_bit         = 0 # Stop bit 
QR_bit           = 0 # QR bit 

# Flag Variable 
i                = 0 # Log variable
w                = 0 # wait variable
q                = 0 # qr variable

try:
	while True:		
		i += 1		
		frames = pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()

		if not color_frame:
		  continue

		# Convert images to numpy arrays
		color_image    = np.asanyarray(color_frame.get_data())	

		# QR Code Check 		
		qr_frame_gray  = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
		decoded_QR     = pyzbar.decode(qr_frame_gray)
						
		canvas      = color_image.copy()*0 	
		canvas2     = canvas.copy() * 0
		canvas_height, canvas_width = canvas.shape[:2]
				
		
		# Camera Adaptataion 
		if w > 50:
			Camera_init = True

		if Camera_init == False:			
			w += 1
			cv2.imshow("Color", color_image)
			cv2.waitKey(1)
			continue
		
		# ############################## Wait ######################### #        
		if QR_bit == 1 and q < 50:
			q += 1
			print(f"Code : {decoded_direction} || q value : {q} || I Value {i}")

			Left_Wheel  = int(Left_Wheel)
			Right_Wheel = int(Right_Wheel)

			if Left_Wheel <= 0:
				Send_Left_Wheel = str("000")
			elif Left_Wheel < 10:
				Send_Left_Wheel = str("00") + str(Left_Wheel)
			elif Left_Wheel < 100:			
				Send_Left_Wheel = str("0") + str(Left_Wheel)
			else:
				Send_Left_Wheel = str(Left_Wheel)

			if Right_Wheel <= 0:
				Send_Right_Wheel = str("000")
			elif Right_Wheel < 10:
				Send_Right_Wheel = str("00") + str(Right_Wheel)
			elif Right_Wheel < 100:			
				Send_Right_Wheel = str("0") + str(Right_Wheel)
			else:
				Send_Right_Wheel = str(Right_Wheel)		

			# UART
			if i % 5 == 0: 					
				print(f"Left Wheel : {Send_Left_Wheel} || Right_Wheel : {Send_Right_Wheel}")
				logger.info(f"START,{Stop_bit},R,{Send_Right_Wheel},D,{Right_direction},L,{Send_Left_Wheel},D,{Left_direction},Z")

			continue

		elif QR_bit == 1 and q >= 50:
			QR_bit = 0 
			q      = 0                
			sys.exit()                   # ------------------------------------------- debugging .... ---------------------------- #
			Init_distance = True		

		# QR Code Part
		if len(decoded_QR) != 0: # QR CODE DETECTED !! 				
			for d in decoded_QR:				
				decoded_direction = d.data.decode('utf-8') # decoded_direction -> L | R | F | S
				cv2.rectangle(color_image, (d.rect[0],d.rect[1]), (d.rect[0] + d.rect[2], d.rect[1]+d.rect[3]), (0,0,255), 2)					

			# QR Size Check 
			if (d.rect[0] + d.rect[2] > 500) or (d.rect[1]+d.rect[3] > 500):
				QR_bit = 1
			else:
				QR_bit = 0 									

			if QR_bit == 1:
				# Use QR_bit  / q variable         
				if decoded_direction == 'R':  # Right
					Left_Wheel      = 100
					Right_Wheel     = 100
					Left_direction  = 1
					Right_direction = 0	

				elif decoded_direction == 'L': # Left 			
					Left_Wheel      = 100
					Right_Wheel     = 100
					Left_direction  = 0
					Right_direction = 1	
				
				elif decoded_direction == 'U': # Slow					
					Left_Wheel      = 100
					Right_Wheel     = 100
					Left_direction  = 1
					Right_direction = 1	

				elif decoded_direction == 'S': # Fast 					
					Left_Wheel      = 0
					Right_Wheel     = 0
					Left_direction  = 1
					Right_direction = 1

				else:
					print("decoded QR Code is wrong ... ")
				
				cv2.imshow("Color", color_image)
				cv2.waitKey(1)                

			else:
				pass

		else:
			pass

finally:    
    # Stop streaming
    pipeline.stop()
