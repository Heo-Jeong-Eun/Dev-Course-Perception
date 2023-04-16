'''
프로그램 흐름도 

1. 카메라 노드가 보내는 토픽에서 영상 프레임 획득
2. 카메라 Calibration 설정값으로 이미지 보정
3. 원근 변환으로 차선 이미지를 Bird's Eye View로 변환
4. OpenCV 영상 처리
    - Gaussian Blur, 노이즈 제거
    - cvtColor, BGR을 HLS 포맷으로 변경
    - threshold, 이진화 처리 
5. 히스토그램을 사용해서 좌우 차선의 시작 위치 파악
6. 슬라이딩 윈도우를 좌우 9개씩 쌓아 올리기
7. 왼쪽과 오른쪽 차선의 위치 찾기
8. 적절한 조향값 계산하고 모터 제어 토픽 발행 
'''

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

# 영상 사이즈에서 ROI 영역만 잘라서 사용한다. 
# ROI 영역 = 세로 480에서 420-460, 즉 40 픽셀만큼만 잘라서 사용한다.
Width = 640
Height = 480
Offset = 420
Gap = 40

motor_control = xycar_motor()

# usb_cam 연결
bridge = CvBridge()
cv_image = np.empty(shape = [0])

# 카메라 영상 title 
window_title = 'camera'

# 영상 불러오기 
def image_callback(data):
        global cv_image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

# 선분 그리기
# 허프 변환 함수로 검출된 모든 선분을 다양한 색깔로 출력한다. 
def draw_lines(image, lines):
    global Offset
    
    # loop를 돌며 시작점과 끝점을 찾고 random하게 색을 입혀 선을 긋는다. 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        
        # 허프 변환에서 사용하는 관심 영역에 넣기 위해 offset을 더해준다. 
        image = cv2.line(image, (x1, y1 + Offset), (x2, y2 + Offset), color, 2)
    
    return image

# 사각형 그리기 
def draw_rectangle(image, lpos, rpos, offset = 0):
    # 중앙 = 왼쪽 + 오른쪽을 2로 나눈 값
    center = (lpos + rpos) / 2

    # lpos 위치에 녹색 사각형 그리기
    cv2.rectangle(image, (lpos - 5, 15 + offset), 
                        (lpos + 5, 25 + offset),
                        (0, 255, 0), 2)
    
    # rpos 위치에 녹색 사각형 그리기 
    cv2.rectangle(image, (rpos - 5, 15 + offset), 
                        (rpos + 5, 25 + offset),
                        (0, 255, 0), 2)    
    
    # lpos, rpos 사이에 녹색 사각형 그리기 
    cv2.rectangle(image, (center - 5, 15 + offset), 
                        (center + 5, 25 + offset),
                        (0, 255, 0), 2)

    # 화면 중앙에 빨간 사각형 그리기 -> 차의 중앙 
    # 0-640의 중앙은 320이므로 -5, +5를 해주어 중앙을 찾는다. 
    cv2.rectangle(image, (315, 15 + offset), 
                        (325, 25 + offset),
                        (0, 0, 255), 2)

    return image

# 왼쪽 선분, 오른쪽 선분
def divide_left_right(lines):
    global Width
    low_slope_threshold = 0
    high_slope_threshold = 10

    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)
        
        # 선분의 기울기를 구해, 기울기 절대값이 10 이하인 것만 추출 
        if (abs(slope) < low_slope_threshold) and (abs(slope) > high_slope_threshold):
            slopes.append(slope)
            new_lines.append(line[0])

        # 허프 변환 함수로 검출한 선분들의 기울기를 비교해 왼쪽 차선과 오른쪽 차선을 구분한다. 
        left_lines = []
        right_lines = []

        # OpenCV 좌표계에서는 아래 방향으로 y가 증가하므로 기울기 계산법이 다르다. 
        # 화면의 왼쪽에 있는 선분 중 기울기가 음수인 것들만 수집
        # 화면의 오른쪽에 있는 선분 중에서 기울기가 양수인 것들만 수집
        for j in range(len(slopes)):
            line = new_lines[j]
            slope = slopes[j]
            x1, y1, x2, y2 = line

            if (slope < 0) and (x2 < Width / 2 - 90):
                left_lines.append([line.tolist()])
            elif(slope > 0) and (x1 > Width / 2 + 90):
                right_lines.append([line.tolist()])
        
        return left_lines, right_lines
    
# 선분의 avg, m, b 
def get_line_params(lines):
    # x, y, m의 합
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)

    if size == 0:
        return 0.0
    
    # 허프 변환 함수로 찾아낸 직선을 대상으로 Parameters Space(m, b)에서 m의 평균값을 먼저 구하고, 이 값으로 b를 구한다. 
    # m의 평균값을 구하는 이유 -> 허프 변환 함수의 결과로 하나가 아닌 여러 개의 선이 검출되기 때문이다. 
    # 찾은 선들의 평균값을 이용하기 때문에 m의 평균을 구해야 한다. 
    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)
    
    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size 
    b = y_avg - m * x_avg

    # 선분의 평균 기울기, y절편을 return 한다. 
    return m, b

# lpos, rpos, 차선의 위치를 return 한다. 
def get_line_pos(image, lines, left = False, right = False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)

    # 차선이 인식되지 않으면 left는 0 값을, right는 width(640)값으로 설정 -> 화면의 끝 좌표 
    # 직선의 방정식에서 y = 20을 넣어 x 좌표를 찾는다. 
    # 640 x 480 원본 이미지의 맨 아래의 x1과 이미지 중간의 x2를 구해 (x1, 480)와 (x2, 320) 두 점을 잇는 파란색 선을 그린다. 
    # 원본 이미지 맨 아래 -> y값이 480, 이미지의 중간 -> y값이 240
    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2

        # y = m0x + b0
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height / 2) - b) / float(m)

        # 파란선 출력 
        cv2.line(image, (int(x1), Height), (int(x2), (Height / 2)), (255, 0, 0), 3)
    
    return image, int(pos)

# 카메라 영상 처리 
# show image, return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap

    # gray 색상으로 전환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur, gaussian blur 처리 
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # canny edge, 외곽선 추출
    low_threshold = 60
    high_threshold = 70
    edge_image = cv2.Canny(np.unit8(blur_gray), low_threshold, high_threshold)

    # HoughLineP, ROI 영역에서 선분 찾기 
    # edge_image -> 640 * 480 size, roi에서 offset을 더해 관심 영역만 저장
    roi = edge_image[Offset : Offset + Gap, 0 : Width]

    # HoughLineP(image, rho, theta, theshold, minLineLength, maxLineGap)
    all_lines = cv2.HoughLineP(roi, 1, math.pi / 180, 30, 30, 10)

    # 선분을 왼쪽과 오른쪽으로 분류
    if all_lines is None:
        # 차선이 없는 경우 가로 양 끝점인 0과 640을 return 
        return 0, 640
    
    left_lines, right_lines = divide_left_right(all_lines)

    # 선분의 정보를 받아 이미지에 차선, 위치 구하기
    frame, lpos = draw_lines(frame, left_lines)
    frame, rpos = draw_lines(frame, right_lines)

    # ROI 영역 안에서 허프 변환을 통해 구한 차선을 랜덤한 색상으로 그린다. 
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255, 255, 255), 2)

    # 차선과 화면 중앙에 사각형 그리기 
    frame = draw_rectangle(frame, lpos, rpos, offset = Offset)

    return lpos, rpos

def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    # 조향 이미지 읽어오기 
    arrow_pic = cv2.imread('steer_arrow.png', cv2.INREAD_COLOR)

    # 이미지 축소를 위한 크기 계산 
    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height / 2
    arrow_Width = (arrow_Height * 462) / 728

    # steer_angle에 비례하여 회전
    matrix = cv2.getRotatedMatrix2D((origin_Width / 2, steer_wheel_center), (steer_angle) * 2.5, 0.7)

    # 이미지 크기를 영상에 맞춘다. 
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width + 60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize = (arrow_Width, arrow_Height), interpolation = cv2.INTER_AREA)

    # 전체 그림 위에 핸들 모양의 그림을 오버레이
    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height : Height, 
                      (Width / 2 - arrow_Width / 2) : (Width / 2 + arrow_Width / 2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask = mask)

    res = cv2.add(arrow_roi, arrow_pic)

    # ? 
    image[(Height - arrow_Height) : Height,
          (Width / 2 - arrow_Width / 2) : (Width / 2 + arrow_Width / 2)] = res
    
    # 'steer' 타이틀로 화면에 표시
    # 원본 사진 + 검출 차선 + 평균 차선 + 차선 위치 표시 + 화면 중앙 표시 핸들 그림 + 조향각 화살표 표시 
    cv2.imshow('steer', image)

# 카메라 토픽 받아오기 
rospy.init_node('cam_tune', anonymous = True)
rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)

# 모터 토픽 발행 
pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size = 1)
rate = rospy.Rate(20)

def pub_motor(angle, speed):
    global pub
    global motor_control

    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

# 시작점
def start():
    global image, Width, Height

    while not rospy.is_shutdown():
        # 허프 변환을 기반으로 영상 처리 진행, 차선을 찾고 위치 표시
        pos, frame = process_image(image)

        # 왼쪽과 오른 쪽 차선의 중간점과 화면 중앙의 차이를 가지고 핸들 조향각을 결정해서 핸들 그림 표시  
        center = (pos[0] + pos[1]) / 2
        angle = 320 - center
        steer_angle = angle * 0.4
        draw_steer(frame, steer_angle)

        # 모터 제어 토픽을 읽어오기 
        Angle = 0
        Speed = 5

        pub_motor(Angle, Speed)

        # 종료 
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    
    #cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start()