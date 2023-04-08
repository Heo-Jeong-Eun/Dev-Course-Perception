#!/usr/bin/env python

import rospy, time
import numpy as np
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor

# 라이다 센서 거리 정보를 담을 저장 공간 준비
motor_msg = xycar_motor()
distance = None

# 라이다 센서 토픽이 들어오면 실행되는 콜백 함수 정의
def callback(data):
    global distance
    distance = np.array(data.ranges)

# 전진 
def drive_go():
    global motor_msg
    motor_msg.speed = 5
    motor_msg.angle = 0
    pub.publish(motor_msg)

# 정지 
def drive_stop():
	global motor_msg
	motor_msg.speed = 0
	motor_msg.angle = 0
	pub.publish(motor_msg)

# 오른쪽으로 회전 
def drive_spin_r():
    global motor_msg
    motor_msg.speed = 3
    motor_msg.angle = 45
    pub.publish(motor_msg)

# 왼쪽으로 회전 
def drive_spin_l(): 
    global motor_msg
    motor_msg.speed = 3
    motor_msg.angle = -45
    pub.publish(motor_msg)   

# 노드 선언, 구독과 발행할 토픽 선언
rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
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

    try:
        # 값이 여러개가 있기 때문에 그 중 최소값만 찾아서 저장한다. 
        min_front = min(front_range)
        min_right = min(right_range)
        min_left = min(left_range)

        # 최소 기준 거리 
        min_distance = 0.3

        # 최소 거리 이내 장애물이 있는 경우 정지 
        if min_front < min_distance:
            drive_stop()
            break
        # 최소 거리 이내, 오른쪽에 장애물이 있을 때 왼쪽으로 회전 
        elif min_right < min_distance:
            drive_spin_l()
            time.sleep(0.15) 
        # 최소 거리 이내, 왼쪽에 장애물이 있을 때 오른쪽으로 회전 
        elif min_left < min_distance:
            drive_spin_r() 
            time.sleep(0.15)      
        # 이상 없는 경우 직진, 왼쪽 + 오른쪽에 아무것도 없는 경우 포함
        else:
            drive_go()
            time.sleep(0.15) 
    except:
        pass