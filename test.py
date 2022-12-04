import cv2, numpy as np, imutils
from stacking import stackImages
camera = cv2.VideoCapture(0)

def convert_rgb_to_hsv(ls: list):
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

	# hue /= 2
	# saturation *= 255
	# value *= 255

	return hue, saturation, value

MAX = np.array([140,255,255])
MIN = np.array([100,150,0])
# MAX = np.array(convert_rgb_to_hsv([0, 0, 255]))
# MIN = np.array(convert_rgb_to_hsv([4, 18, 96]))

while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	frame = imutils.resize(frame, width=600)

	## convert to hsv
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	## mask of blue
	mask = cv2.inRange(hsv, MIN, MAX)
	mask = cv2.erode(mask, None, iterations=4)
	mask = cv2.dilate(mask, None, iterations=4)

	## final mask and masked
	target = cv2.bitwise_and(frame, frame, mask=mask)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		cv2.drawContours(target, cnts,-1,(0,255,0),3)

		# draw the circle and centroid on the frame
		cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
		cv2.circle(frame, center, 5, (255,255,0), -1)

	# loop over each of the individual channels and display them
	h, s, v = cv2.split(hsv)
	cv2.imshow("stack", stackImages([[frame, target, mask], [h, s, v]], 0.1))

	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break