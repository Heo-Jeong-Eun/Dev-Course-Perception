#!/usr/bin/env python

import rospy, time
import numpy as np
import tf
from sensor_msgs.msg import LaserScan
# from xycar_msgs.msg import xycar_motor

# motor_msg = xycar_motor()
distance = None

def callback(data):
	global distance
	distance = data.ranges
	# distance = np.array(data.ranges)

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

def drive_spin_r():
	global motor_msg
	motor_msg.speed = 3
	motor_msg.angle = 45
	pub.publish(motor_msg)

def drive_spin_l(): 
	global motor_msg
	motor_msg.speed = 3
	motor_msg.angle = -45
	pub.publish(motor_msg)   

rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
# pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

while not rospy.is_shutdown():
	print(distance)
	# front_range = distance[((distance.size * 3) // 8) : ((distance.size * 5) // 8)]
	# right_range = distance[((distance.size * 6) // 8) : ((distance.size * 7) // 8)]
	# left_range = distance[((distance.size * 1) // 8) : ((distance.size * 2) // 8)]

	# front_range = front_range[front_range != 0]
	# right_range = right_range[right_range != 0]
	# left_range = left_range[left_range != 0]

	# try:
	# print("test")
	# min_front = min(front_range)
	# min_right = min(right_range)
	# min_left = min(left_range)

	# min_distance = 0.3

	'''
	if min_front < min_distance:
	drive_stop()
	break
	elif min_right < min_distance:
	drive_spin_l()
	time.sleep(0.15) 
	elif min_left < min_distance:
	drive_spin_r() 
	time.sleep(0.15)      
	else:
	drive_go()
	time.sleep(0.15) 
	'''

	# print(front_range)
	# print(right_range)
	# print(left_range)

	# print("f" + min_front)
	# print("r" + min_right)
	# print("l" + min_left)

	# except:
	# 	pass
