#!/usr/bin/env python

import rospy
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor
import logging

# setting logger, log를 사용하기 위한 초기 설정 
logger = logging.getLogger(__name__) # logger 생성, 기본값, 지정하면 특정 파일만 적용 가능 
logger.setLevel(logging.DEBUG) # DEBUG Level부터 handler에게 전달
formatter = logging.Formatter('%(asctime)s -- [%(levelname)s]\n%(message)s\n') # 로그 메시지 format 설정

# setting logger-handler
stream_handler = logging.StreamHandler() # 스트림으로 로그를 출력하는 handler(stream_handler) 생성
stream_handler.setLevel(logging.INFO) # console에서 INFO Level부터 표시
stream_handler.setFormatter(formatter) # 위에서 지정한 formatter 형식으로 handler format 설정
logger.addHandler(stream_handler) # stream_handler 객체 추가

# 라이다 센서 거리 정보를 담을 저장 공간 준비
motor_msg = xycar_motor()
distance = None

# 모터 메시지를 발행하는 함수
def publish_motor_msg(speed, angle):  
    # log msg - info
    logger.info('-- START: publish_motor_msg() --')
    
    global motor_msg
    motor_msg.speed = speed
    motor_msg.angle = angle
    pub.publish(motor_msg)

# 라이다 센서 토픽이 들어오면 실행되는 콜백 함수 정의
def lidar_callback(data):
    global distance
    distance = np.array(data.ranges) 

def start():
    # log msg - info
    logger.info('-- START: start() --')
    
    # 주기 설정, 0.15
    rate = rospy.Rate(0.15)

    # 노드 선언, 구독과 발행할 토픽 선언
    rospy.init_node('lidar_driver')
    rospy.Subscriber('/scan', LaserScan, lidar_callback, queue_size = 1)
    pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

    while not rospy.is_shutdown():

        # 라이다 센서를 8등분 하고, 앞, 왼, 오를 슬라이싱 해 저장한다.
        front_range = distance[((distance.size * 3) // 8) : ((distance.size * 5) // 8)]
        right_range = distance[((distance.size * 6) // 8) : ((distance.size * 7) // 8)]
        left_range = distance[((distance.size * 1) // 8) : ((distance.size * 2) // 8)]

        # 0이 나오는 경우는 모두 제외, 불필요한 0 값이 너무 많아서 제거한다. 
        front_range = front_range[front_range != 0]
        right_range = right_range[right_range != 0]
        left_range = left_range[left_range != 0]

        # 값이 여러개가 있기 때문에 그 중 최소값만 찾아서 저장한다.
        min_front = min(front_range)
        min_right = min(right_range)
        min_left = min(left_range)

        # 최소 기준 거리
        min_distance = 0.3

        rate.sleep() 
        
        # 최소 거리 이내 장애물이 있는 경우 정지
        if min_front < min_distance:
            publish_motor_msg(0, 0)
            break 
        # 최소 거리 이내, 오른쪽에 장애물이 있을 때 왼쪽으로 회전 
        elif min_right < min_distance:
            publish_motor_msg(3, -45) 
        # 최소 거리 이내, 왼쪽에 장애물이 있을 때 오른쪽으로 회전 
        elif min_left < min_distance:
            publish_motor_msg(3, 45)     
        # 이상 없는 경우 직진, 왼쪽 + 오른쪽에 아무것도 없는 경우 포함
        else:
            publish_motor_msg(5, 0)

# 시작점 지정
if __name__ == '__main__':
    try:
        start()
    except:
        # log msg - error
        logger.error('SOMETHING WAS WRONG...')