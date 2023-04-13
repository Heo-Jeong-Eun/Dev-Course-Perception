import rospy
import cv2
import numpy as np
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
img = np.empty(shape = [0])

def img_callback(data):
    global img
    img = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('cam_tune', anonymous = True)
rospy.Subscriber('/usb_cam/image_raw', Image, img_callback)

# img = cv2.imread('')
rows, cols = img.shape[0:2]

pts1 = np.float32([[20, 20], [20, 280], [380, 20], [380, 280]])
pts2 = np.float32([[100, 20], [20, 280], [300, 20], [380, 280]])

cv2.circle(img, (20, 20), 20, (255, 0, 0), -1)
cv2.circle(img, (20, 280), 20, (0, 255, 0), -1)
cv2.circle(img, (380, 20), 20, (0, 0, 255), -1)
cv2.circle(img, (380, 280), 20, (0, 255, 255), -1)

M = cv2.getPerspectiveTransform(pts1, pts2)
print M

dst = cv2.warpPerspective(img, M, (400, 300))

plt.subplot(121), plt.imshow(img), plt.title('image')
plt.subplot(122), plt.imshow(dst), plt.title('Perspective')
plt.show()
