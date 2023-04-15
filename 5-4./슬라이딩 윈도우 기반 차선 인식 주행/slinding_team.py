#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy
import rospy
from xycar_motor.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Variables
bridge = CvBridge()
motor_control = xycar_motor()

image = np.empty(shape = [0])
image_width, image_height = 640, 480
image_area = image_width * image_height
warp_image_width = 320
warp_image_height = 240

num_sliding_window = 9
area_sliding_window = 12
mininum_points = 5

warp_x_margin = 50
warp_y_margin = 10

lane_bin_th = 145

pts1_x, pts1_y = 290 - warp_x_margin, 290 - warp_y_margin
pts2_x, pts2_y = 100 - warp_x_margin, 410 + warp_y_margin
pts3_x, pts3_y = 440 + warp_x_margin, 290 - warp_y_margin
pts4_x, pts4_y = 580 + warp_x_margin, 400 + warp_y_margin

warp_src = np.array([
    [pts1_x, pts1_y],
    [pts2_x, pts2_y],
    [pts3_x, pts3_y],
    [pts4_x, pts4_y]
], dtype = np.float32)

warp_dist = np.array([
    [0, 0],
    [0, warp_image_height],
    [warp_image_width, 0],
    [warp_image_width, warp_image_height]
], dtype = np.float32)

# Class
class PID():
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte): # CTE = 화면의 중앙값과 좌우차선의 중점과의 차이
        self.d_error=cte-self.p_error
        self.p_error=cte
        self.i_error += cte

        return self.Kp*self.p_error + self.Ki*self.i_error + self.Kd*self.d_error

# Function
def image_callback(data):
    global image    
    image = bridge.imgmsg_to_cv2(data, "bgr8")

def publish_motor_control_msg(angle, speed):
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)  

def calibrate_image(src):
    global image_width, image_height

    calibrate_mtx = np.array([
    [ 350.354184, 0.0, 328.104147],
    [0.0, 350.652653, 236.540676],
    [0.0, 0.0, 1.0]
    ])

    dist = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])

    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(calibrate_mtx, dist,
                    (image_width, image_height), 1, (image_width, image_height))

    dst = cv2.undistort(src, calibrate_mtx, dist, None, cal_mtx)
    x, y, w, h = cal_roi
    calibrated_image = dst[y : y + h, x : x + w]

    return cv2.resize(calibrated_image, (image_width, image_height))

def warp_image(img, src, dst, size):
    src_to_dst_mtx = cv2.getPerspectiveTransform(src, dst)
    dst_to_src_mtx = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, src_to_dst_mtx, size, flags = cv2.INTER_LINEAR)
    return warp_img, src_to_dst_mtx, dst_to_src_mtx

def warp_process_image(image):
    global num_sliding_window
    global area_sliding_window
    global mininum_points

    # Gaussian Blurring
    blur = cv2.GaussianBlur(image, (5, 5), 0) 

    # detect white
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))  

    # lane_bin_th: threshold
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY_INV) 

    histogram = np.sum(lane[lane.shape[0] // 2 :, :], axis = 0)

    midpoint = np.int(histogram.shape[0] / 2)

    leftx_current = np.argmax(histogram[:midpoint])

    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(lane.shape[0] / num_sliding_window)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    lx, ly, rx, ry = [], [], [], []
    out_img = np.dstack((lane, lane, lane)) * 255

    l_box_center, r_box_center = [], []

    for window in range(num_sliding_window):
        win_yl = lane.shape[0] - (window + 1) * window_height
        win_yh = lane.shape[0] - window * window_height
    
        win_xll = leftx_current - area_sliding_window
        win_xlh = leftx_current + area_sliding_window
        win_xrl = rightx_current - area_sliding_window
        win_xrh = rightx_current + area_sliding_window
    
        cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)
    
        good_left_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xll) & (nz[1] < win_xlh)).nonzero()[0]
    
        good_right_inds = ((nz[0] >= win_yl) & (nz[0] < win_yh) & (nz[1] >= win_xrl) & (nz[1] < win_xrh)).nonzero()[0]
    
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        if len(good_left_inds) > mininum_points:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
            cv2.rectangle(out_img, (win_xll, win_yl), (win_xlh, win_yh), (0, 255, 0), 2)
            l_box_center.append([(win_xll + win_xlh) // 2, (win_yl + win_yh) // 2])
        if len(good_right_inds) > mininum_points:
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))
            cv2.rectangle(out_img, (win_xrl, win_yl), (win_xrh, win_yh), (0, 255, 0), 2)
            r_box_center.append([(win_xrl + win_xrh) // 2, (win_yl + win_yh) // 2])

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

    return lfit, rfit, np.mean(lx), np.mean(rx), out_img, l_box_center, r_box_center

def draw_lane(image, warp_img, Minv, left_fit, right_fit):
    global image_width, image_height

    yMax = warp_img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(warp_img).astype(np.uint8)

    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))

    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_width, image_height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)

def get_line_pos(lst):
    global image_width

    low_slope_threshold = 0
    high_slope_threshold = 100

    slopes = []
    new_lines = []
    x1 = lst[0][0]
    x2 = lst[-1][0]
    y1 = lst[0][1]
    y2 = lst[-1][1]

    dx = x2 - x1

    if dx == 0:
        dx = 0.01
    slope = float(y2-y1) / dx
    if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
        return slope
    else:
        return float("inf")

def check_middle(lx, rx):
    l_x = lx[3]
    r_x = rx[3]

    if l_x is None:
        l_x = 0

    if r_x is None:
        r_x = 320

    middle = (l_x + r_x)/2

    return middle

def start():
    global pub
    global image
    global image_width, image_height

    while True:
        while not image.size == (image_area * 3):
            continue

        calibrated_image = calibrate_image(image)
        warp, src_mtx, dst_mtx = warp_image(calibrated_image, warp_src, warp_dist, (warp_image_width, warp_image_height))
        l_fit, r_rit, l_mean, r_mean, out_img, l_box_center, r_box_center = warp_process_image(warp)
        
        l_slope = get_line_pos(l_box_center)
        r_slope = get_line_pos(r_box_center)
        
        if l_slope + r_slope == 0:
            angle = 0
        else:
            angle = -(1 / ((l_slope + r_slope) / 2)) * 3
        
        # image_middle = 160
        # real_middel = check_middle(lx, rx)
        pid = PID(0.5, 0.0005, 0.05)
        # error =  -real_middel + image_middle
        # angle = (pid.pid_control(error))
        print(angle)

        # if angle > 50:
        #     angle = 50
        # elif angle < -50:
        #     angle = -50
        #print("l: %.1f r: %.1f angle : %.1f" %(l_slope, r_slope, angle))

        publish_motor_control_msg(angle, 3)

        # cv2.imshow("src", image)
        # cv2.imshow("cal", calibrated_image)
        # cv2.imshow("warp points", warp)
        # cv2.imshow("bird eye view", out_img)

# Setting
rospy.init_node('driver')
rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

rospy.Rate(10)

# main
if __name__ == '__main__':
    start()