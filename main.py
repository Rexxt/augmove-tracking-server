import json
import numpy as np
import imutils
import cv2
import time
from stacking import stackImages
import hug
from falcon import get_http_status

""" def convert_rgb_to_hsv(ls: list):
	red, green, blue = ls[0], ls[1], ls[2]
	#get rgb percentage: range (0-1, 0-1, 0-1 )

	value = max(red, green, blue)
	if value != 0:
		saturation = (value-min(red, green, blue))/value
	else:
		saturation = 0
	
	if value == red:
		hue = 60*(green-blue)/(value-min(red, green, blue))
	elif value == green:
		hue = 120 + 60*(blue-red)/(value-min(red, green, blue))
	elif value == blue:
		hue = 240 + 60*(red-green)/(value-min(red, green, blue))
	
	if hue < 0:
		hue += 360

	hue /= 2
	saturation *= 255
	value *= 255

	return hue, saturation, value """

config = json.load(open('config.json', 'r'))
controllers = config['hsv_controller_colours']
tracked_controllers = {}

print(config, controllers)

camera = cv2.VideoCapture(0)

for con in controllers:
	print(con)
	for k in con:
		print(k)
		for l in con[k]:
			print(l)
			con[k][l] = np.array(con[k][l])

@hug.get('/track-controllers')
def track_controllers(response, show: hug.types.boolean = False):
	global camera, tracked_controllers, controllers
	# grab the current frame
	(grabbed, frame) = camera.read()

	if not grabbed:
		response.status = get_http_status(500)
		print('frame not grabbed')
		return {'error': 'frame not grabbed'}

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	tracked_controllers = {}
	for i in range(len(controllers)):
		con = controllers[i]

		print(con)

		height, width, channels = frame.shape

		mask1 = cv2.inRange(hsv, con['L']['lower'], con['L']['upper'])
		mask1 = cv2.erode(mask1, None, iterations=3)
		mask1 = cv2.dilate(mask1, None, iterations=3)

		mask2 = cv2.inRange(hsv, con['R']['lower'], con['R']['upper'])
		mask2 = cv2.erode(mask2, None, iterations=3)
		mask2 = cv2.dilate(mask2, None, iterations=3)

		target = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(mask1, mask2))

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
				# draw the circle and centroid on the frame
				cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
				cv2.circle(frame, center1, 5, (255,255,0), -1)

				if not i in tracked_controllers:
					tracked_controllers[i] = [None, None]

				# update controller position
				tracked_controllers[i][0] = [x, y, radius]
		
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
				cv2.circle(frame, (int(x), int(y)), int(radius), (255,0,255), 2)
				cv2.circle(frame, center2, 5, (0,128,255), -1)

				if not i in tracked_controllers:
					tracked_controllers[i] = [None, None]

				# update controller position
				tracked_controllers[i][1] = [x, y, radius]

	print(tracked_controllers)

	# show the frame to our screen
	if show:
		cv2.imshow("Frame", stackImages([[frame, target], [mask1, mask2]], 0.5, labels=[['', f'Con {len(controllers)-1} Frame & (Mask1|Mask2)'], [f'Con {len(controllers)-1} L', f'Con {len(controllers)-1} R']]))
	camera.release()
	return tracked_controllers

""" # keep looping
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
	
	tracked_controllers = {}
	for i in range(len(controllers)):
		con = controllers[i]

		print(con)

		height, width, channels = frame.shape

		mask1 = cv2.inRange(hsv, con['L']['lower'], con['L']['upper'])
		mask1 = cv2.erode(mask1, None, iterations=3)
		mask1 = cv2.dilate(mask1, None, iterations=3)

		mask2 = cv2.inRange(hsv, con['R']['lower'], con['R']['upper'])
		mask2 = cv2.erode(mask2, None, iterations=3)
		mask2 = cv2.dilate(mask2, None, iterations=3)

		target = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_or(mask1, mask2))

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
				# draw the circle and centroid on the frame
				cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
				cv2.circle(frame, center1, 5, (255,255,0), -1)

				if not i in tracked_controllers:
					tracked_controllers[i] = [None, None]

				# update controller position
				tracked_controllers[i][0] = [x, y, radius]
		
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
				cv2.circle(frame, (int(x), int(y)), int(radius), (255,0,255), 2)
				cv2.circle(frame, center2, 5, (0,128,255), -1)

				if not i in tracked_controllers:
					tracked_controllers[i] = [None, None]

				# update controller position
				tracked_controllers[i][1] = [x, y, radius]

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
	cv2.imshow("Frame", stackImages([[frame, target], [mask1, mask2]], 0.5, labels=[['', f'Con {len(controllers)-1} Frame & (Mask1|Mask2)'], [f'Con {len(controllers)-1} L', f'Con {len(controllers)-1} R']]))
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break """

# cleanup the camera and close any open windows
cv2.destroyAllWindows()