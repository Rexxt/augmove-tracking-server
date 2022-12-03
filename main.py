import json
import numpy as np
import argparse
import imutils
import cv2
import time
import colorsys

def convert_rgb_to_hsv(ls: list):
	red, green, blue = ls[0], ls[1], ls[2]
	#get rgb percentage: range (0-1, 0-1, 0-1 )

	red_percentage= red / float(255)
	green_percentage= green / float(255)
	blue_percentage=blue / float(255)
	
	#get hsv percentage: range (0-1, 0-1, 0-1)
	color_hsv_percentage=colorsys.rgb_to_hsv(red_percentage, green_percentage, blue_percentage) 

	#get normal hsv: range (0-360, 0-255, 0-255)
	color_h=round(179*color_hsv_percentage[0])
	color_s=round(255*color_hsv_percentage[1])
	color_v=round(255*color_hsv_percentage[2])
	color_hsv=(color_h, color_s, color_h)

	return np.array([color_h, color_s, color_h], np.uint8)

config = json.load(open('config.json', 'r'))
controllers = config['hsv_controller_colours']
tracked_controllers = {}

print(config, controllers)

camera = cv2.VideoCapture(0)

# start time
start_time = time.time()

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

# frames
frames = 0

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	if not grabbed:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	tracked_controllers = []
	for i in range(len(controllers)):
		con = controllers[i]
		tracked_controllers = {}

		print(con)

		mask1 = cv2.inRange(hsv, convert_rgb_to_hsv(con['L']['lower']), convert_rgb_to_hsv(con['L']['upper']))
		cv2.imshow("mask1", mask1)
		mask1 = cv2.erode(mask1, None, iterations=2)
		mask1 = cv2.dilate(mask1, None, iterations=2)

		mask2 = cv2.inRange(hsv, convert_rgb_to_hsv(con['R']['lower']), convert_rgb_to_hsv(con['R']['upper']))
		cv2.imshow("mask2", mask2)
		mask2 = cv2.erode(mask2, None, iterations=2)
		mask2 = cv2.dilate(mask2, None, iterations=2)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		center1 = None

		# only proceed if at least one contour was found
		if len(cnts1) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts1, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					con['L']['upper'], 2)
				cv2.circle(frame, center, 5, con['L']['lower'], -1)

				# update controller position
				tracked_controllers[i][0] = [x, y]
		
		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		center2 = None

		# only proceed if at least one contour was found
		if len(cnts2) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts2, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					con['R']['upper'], 2)
				cv2.circle(frame, center, 5, con['R']['lower'], -1)

				# update controller position
				tracked_controllers[i][1] = [x, y]

	frames += 1
	# time when we finish processing for this frame
	new_frame_time = time.time()

	print(tracked_controllers)

	# font which we will be using to display FPS
	font = cv2.FONT_HERSHEY_SIMPLEX

	fps = str(int(1/(new_frame_time-prev_frame_time)))
	avg_fps = frames/(time.time()-start_time)
	avg_fps = str(round(avg_fps))
	prev_frame_time = new_frame_time

	# putting the FPS count on the frame
	cv2.putText(frame, fps, (0, 25), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
	cv2.putText(frame, avg_fps, (40, 25), font, 1, (0, 128, 255), 3, cv2.LINE_AA)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()