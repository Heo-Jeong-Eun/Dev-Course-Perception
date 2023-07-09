#!/usr/bin/env python

import numpy as np
import cv2
import math
import rospy

from std_msgs.msg import String
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes

mapping_image = np.uint8([0])

class_dict = {
    0: 'left',
    1: 'right',
    2: 'stop',
    3: 'crosswalk',
    4: 'small',
    5: 'traffic_light',
    6: 'xycar',
    7: 'ignore'
}


def bbox_callback(data):
    global mapping_image
    mapping_image = np.full((540, 540, 3), 255, dtype=np.uint8)
    for i in range(1, 6):
        cv2.line(mapping_image, (90 * i, 0), (90 * i, 540), (0, 0, 255))
        cv2.line(mapping_image, (0, 90 * i), (540, 90 * i), (0, 0, 255))

    for bbox in data.bounding_boxes:
        x = np.float32(bbox.x)
        y = np.float32(bbox.y)
        distance = np.float32(bbox.distance)

        p = [int(200 * x), int(200 * (y + 1.35))]
        cv2.circle(mapping_image, (p[1], p[0]), 5, (255, 0, 0), 2)
        cv2.puText(mapping_image, class_dict[int(bbox.id)], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Mapping View", mapping_image)
    cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('mapping_view', annoymous=True)
    rospy.Subscriber("/yolov3_trt_ros/detections", BoundingBoxes, bbox_callback)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        rate.sleep()
