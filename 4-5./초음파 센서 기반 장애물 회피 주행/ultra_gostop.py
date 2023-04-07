#!/usr/bin/env python

# Int32MultiArray, xycar_motor 메시지 사용 준비 
import rospy, time
from sensor_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor

# 초음파 센서 거리 정보를 담을 저장 공간 준비
ultra_msg = None
motor_msg = xycar_motor()

# 초음파 센서 토픽이 들어오면 실행되는 콜백 함수 정의
def callback(data):
	global ultra_msg
	ultra_msg = data.data

# 전진
def drive_go():
	global motor_msg, pub
	motor_msg.speed = 5
	motor_msg.angle = 0
	pub.publish(motor_msg)

# 정지
def drive_stop():
	global motor_msg, pub
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
rospy.init_node('ultra_driver')
rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, callback, queue_size = 1)
pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size = 1)

# 초음파 센서가 가동할 때까지 잠시 기다린다. 
time.sleep(2) 

while not rospy.is_shutdown():
	# 전방 초음파 센서가 감지한 거리 정보가 0 < 거리 < 10cm 범위에 있으면 정차, 그보다 크면 전진
	# 거리값 0 -> 무한대, 장애물이 없음을 의미한다. 
	# 1 -> 왼쪽
	# 2	-> 오른쪽
	# 3, 4, 5 -> 뒤 / 3 -> 왼쪽 뒤, 5 -> 오른쪽 뒤 

	# 왼쪽 초음파 센서, 왼쪽에 장애물이 있는 경우 
	if ultra_msg[1] > 0 and ultra_msg[1] < 10:
		drive_stop()
		drive_spin_r()

	# 오른쪽 초음파 센서, 오른쪽에 장애물이 있는 경우 
	if ultra_msg[2] > 0 and ultra_msg[2] < 10:
		drive_stop()
		drive_spin_l()

	# 뒷쪽 초음파 센서, 뒤에 장애물이 있는 경우 	
	if ultra_msg[3] > 0 and ultra_msg[3] < 10:
		drive_stop()
	if ultra_msg[4] > 0 and ultra_msg[4] < 10:
		drive_stop()
	if ultra_msg[5] > 0 and ultra_msg[5] < 10:
		drive_stop()
	else:
		drive_go()