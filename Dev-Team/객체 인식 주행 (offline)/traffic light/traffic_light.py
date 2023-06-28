import cv2
import numpy as np

image = cv2.imread('/Users/1001l1000/Desktop/traffic light/image/img3121_xycar.jpg')
#  image = cv2.imread('/Users/1001l1000/Desktop/traffic light/image/img495_sign.jpg')

# blur, median
blur_image = cv2.medianBlur(image, 5)

# BGR -> Gray 
# image2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# BGR -> HSV
hsv = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# detect circle 
circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 25, minRadius = 15, maxRadius = 20)
circles = np.uint16(np.around(circles))

# define the lower, upper boundaries for red, green color
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])

green_lower = np.array([45, 90, 90])
green_upper = np.array([100, 255, 255])

for i in circles[0, :]:
    cr_image = v[i[1] - 10 : i[1] + 10, i[0] - 10 : i[0] + 10]
    image_str = 'x : {0}, y : {1}, mean : {2}'.format(i[0], i[1], cr_image.mean())
    print(image_str)
    print('cr_image max', cr_image.max())
    
    # if np.array_equal(cr_image, cr_image.max()):
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red_result = cv2.bitwise_and(image, image, mask = red_mask)

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_result = cv2.bitwise_and(image, image, mask = green_mask)

cv2.imshow('h', h)
cv2.imshow('s', s)
cv2.imshow('v', v)

# cv2.imshow('gray', image2gray)

cv2.imshow('image', image)
cv2.imshow('Red Result', red_result)
cv2.imshow('Green Result', green_result)

cv2.waitKey(0)
cv2.destroyAllWindows()