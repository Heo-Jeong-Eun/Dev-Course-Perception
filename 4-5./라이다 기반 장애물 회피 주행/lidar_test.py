# error ! numpy, size를 사용한 slicing
#!/usr/bin/env python

import rospy
import time
import numpy as np
from sensor_msgs.msg import LaserScan
# from xycar_msgs.msg import xycar_motor

import logging

# motor_msg = xycar_motor()
distance = None

def publish_motor_msg(speed, angle):   
	global motor_msg
	motor_msg.speed = speed
	motor_msg.angle = angle
	pub.publish(motor_msg)

def lidar_callback(data):
	global distance
	distance = np.array(data.ranges) 

# rate = rospy.Rate(0.15)

def start():
	rospy.init_node('lidar_driver')
	rospy.Subscriber('/scan', LaserScan, lidar_callback, queue_size = 1)
	# pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

	while not rospy.is_shutdown():
		print(distance)
		'''
		front_range = distance[((distance.size * 3) // 8) : ((distance.size * 5) // 8)]
		right_range = distance[((distance.size * 6) // 8) : ((distance.size * 7) // 8)]
		left_range = distance[((distance.size * 1) // 8) : ((distance.size * 2) // 8)]

		front_range = front_range[front_range != 0]
		right_range = right_range[right_range != 0]
		left_range = left_range[left_range != 0]

		min_front = min(front_range)
		min_right = min(right_range)
		min_left = min(left_range)
 
		print(front_range)
		print(right_range)
		print(left_range)
		'''

		'''
		min_distance = 0.3

		rate.sleep()	

		if min_front < min_distance:
		publish_motor_msg(0, 0)
		break

		elif min_right < min_distance:
		publish_motor_msg(3, -45)

		elif min_left < min_distance:
		publish_motor_msg(3, 45)     
		else:
		publish_motor_msg(5, 0)
		'''

if __name__ == '__main__':
	start()