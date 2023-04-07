#!/usr/bin/env python

import rospy, time
import numpy as np
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor

motor_msg = xycar_motor()
distance = None

def callback(data):
    global distance
    distance = np.array(data.ranges)

def drive_go():
    global motor_msg
    motor_msg.speed = 5
    motor_msg.angle = 0
    pub.publish(motor_msg)

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

rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

time.sleep(3) 

while not rospy.is_shutdown():
    try:
        front_range = distance[((distance.size * 3) // 8) : ((distance.size * 5) // 8)]
        right_range = distance[((distance.size * 6) // 8) : ((distance.size * 7) // 8)]
        left_range = distance[((distance.size * 1) // 8) : ((distance.size * 2) // 8)]
        
        front_range = front_range[front_range != 0]
        right_range = right_range[right_range != 0]
        left_range = left_range[left_range != 0]

        ok = 0

	    # 전방이 90도 이므로 60-120도 범위를 스캔한다. 
        for degree in range(front_range):
            # 장애물과 거리 30cm 이내일 때 
            if distance[degree] <= 0.3:
                ok += 1
            # 범위 안에서 ok가 3보다 커지면, 정지
            if ok > 3:
                drive_stop()
                break
            if ok <= 3:
                drive_go()

                # ok <= 3이고, 왼쪽에 장애물이 있을 때 왼쪽으로 회전 
                for degree in range(right_range):
                    if distance[degree] <= 0.3:
                        drive_spin_l() 

                # ok <= 3이고, 왼쪽에 장애물이 있을 때 오른쪽으로 회전 
                for degree in range(left_range):
                    if distance[degree] <= 0.3:
                        drive_spin_r()
        
        # print("f: %.1f" % min(front_range))
        # print("r: %.1f" % min(right_range))
        # print("l: %.1f" % min(left_range))
        
    except:
        pass