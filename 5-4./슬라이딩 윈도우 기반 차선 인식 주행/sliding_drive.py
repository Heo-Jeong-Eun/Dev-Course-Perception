#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import rospkg
import numpy as np
import cv2
import random
import math
import time
import copy

from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

motor_control = xycar_motor()

bridge = CvBridge()
cv_image = np.empty(shape = [0])

Width = 640
Height = 480

# cap = cv2.VideoCapture('xycar_track1.mp4')
window_title = 'camera'

warp_img_w = 320
warp_img_h = 240

warpx_margin = 20
warpy_margin = 3

nwindows = 9
margin = 12
minpix = 5

lane_bin_th = 145

ptx_x1 = 230
ptx_y1 = 230
ptx_x2 = 25
ptx_y2 = 470
ptx_x3 = 465
ptx_y3 = 280
ptx_x4 = 630
ptx_y4 = 470

warp_src = np.array([
    [ptx_x1, ptx_y1],
    [ptx_x2, ptx_y2],
    [ptx_x3, ptx_y3],
    [ptx_x4, ptx_y4]
], dtype = np.float32)

warp_dist = np.array([
    [0, 0],
    [0, warp_img_h],
    [warp_img_w, 0],
    [warp_img_w, warp_img_h],
], dtype = np.float32)

calibrated = True

if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397],
        [0.0, 435.589734, 163.625535],
        [0.0, 0.0, 1.0]
    ])

    dist = np.array([-0.319089, 0.082498, -0.001147, -0.001638, 0.000000])

    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dist,
                                                      (Width, Height), 1, (Width, Height))

def img_callback(data):
        global cv_image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

def calibrate_image(frame):
    global Width, Height
    global mtx, dist
    global cal_mtx, cal_roi

    tf_image = cv2.undistort(frame, mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y : y + h, x : x + w]

    return cv2.resize(tf_image, (Width, Height))

def warp_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, M, size, flags = cv2.INTER_LINEAR)

    return warp_img, M, Minv

def warp_process_image(img):
    global nwindows
    global margin
    global minpix
    global lane_bin_th

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))

    # lane_bin_th = 145
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY_INV)

    histogram = np.sum(lane[lane.shape[0] // 2 :, :], axis = 0)

    midpoint = np.int(histogram.shape[0] / 2)

    leftx_current = np.argmax(histogram[:midpoint])

    rightx_current = np.argmax(histogram[:midpoint]) + midpoint

    window_height = np.int(lane.shape[0] / nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []
    out_img = np.dstack((lane, lane, lane)) * 255

    for window in range(nwindows):
        win_yl = lane.shape[0] - (window + 1) * window_height
        win_yh = lane.shape[0] - window * window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)

        good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & 
                          (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) &
                           (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh) / 2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh) / 2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(ly), np.array(lx), 2)
    rfit = np.polyfit(np.array(ry), np.array(rx), 2)

    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds], nz[1][right_lane_inds]] = [0, 0, 255]
    cv2.imshow('viewer', out_img)

    return lfit, rfit, lane

def draw_lane(image, warp_img, Minv, left_fit, right_fit):
    global Width, Height
    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))

    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (Width, Height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

rospy.init_node('cam_tune', anonymous = True)
rospy.Subscriber("/usb_cam/image_raw", Image, img_callback)
pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size = 1)
rate = rospy.Rate(20)

def pub_motor(angle, speed):
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

def onChange(pos):
    pass

cv2.namedWindow("Trackbar Windows")
cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.setTrackbarPos("threshold", "Trackbar Windows", lane_bin_th)

def start():
    global Width, Height, lane

    while not rospy.is_shutdown():

        while not cv_image.size == (640 * 480 * 3):
            continue
        
        frame = cv_image
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        lane_bin_th = cv2.getTrackbarPos("threshold", "Trackbar Windows")
        
        image = calibrate_image(frame)

        dot_image = image

        warp_img, M, Minv = warp_image(image, warp_src, warp_dist, (warp_img_w, warp_img_h))

        left_fit, right_fit, lane = warp_process_image(warp_img)
        lane_img = draw_lane(image, warp_img, Minv, left_fit, right_fit)

        cv2.circle(dot_image, (ptx_x1,ptx_y1), 20, (255,0,0), -1)
        cv2.circle(dot_image, (ptx_x2,ptx_y2), 20, (0,255,0), -1)
        cv2.circle(dot_image, (ptx_x3,ptx_y3), 20, (0,0,255), -1)
        cv2.circle(dot_image, (ptx_x4,ptx_y4), 20, (0,0,0), -1)

        cv2.imshow(window_title, lane_img)
        cv2.waitKey(1)
        cv2.imshow("lane", lane)
        cv2.imshow("warp", warp_img)
        cv2.imshow("dot", dot_image)

        Angle = 0
        Speed = 5

        pub_motor(Angle, Speed)

    #cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start()