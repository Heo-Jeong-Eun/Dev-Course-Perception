#!/usr/bin/env python

# LaserScan, xycar_motor 메시지 사용 준비
import rospy, time
from sensor_msgs.msg import LaserScan
from xycar_msgs.msg import xycar_motor

# 라이다 거리 정보를 담을 공간 
motor_msg = xycar_motor()
distance = []

# 라이다 토픽이 들어오면 실행되는 콜백 함수 정의
def callback(data):
	global distance, motor_msg
	distance = data.ranges

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

# 노드 선언, 구독과 발행할 토픽 선언
rospy.init_node('lidar_driver')
rospy.Subscriber('/scan', LaserScan, callback, queue_size = 1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

# Laser가 가동할 때까지 잠시 기다린다. 
time.sleep(3) 

while not rospy.is_shutdown():
	ok = 0

	# 전방이 90도 이므로 60-120도 범위를 스캔한다. 
	for degree in range(60, 120):
		if distance[degree] <= 0.3:
			ok += 1
		# 범위 안에서 ok가 3보다 커지면, 정지
		if ok > 3:
			drive_stop()
			break
	if ok <= 3:
		drive_go()