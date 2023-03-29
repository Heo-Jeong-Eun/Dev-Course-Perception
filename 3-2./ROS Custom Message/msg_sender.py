#!/usr/bin/env python

import rospy
from msg_send.msg import my_msg

rospy.init_node('msg_sender', anonymous=True)
pub = rospy.Publisher('msg_to_xycar', my_msg, queue_size=1)

msg = my_msg()
msg.first_name = "Gil-Dong"
msg.last_name = "Hong"
msg.id_number = 20230329
msg.phone_number = "010-1234-5678"

rate = rospy.Rate(1)

while not rospy.is_shutdown():
	pub.publish(msg)
	print("sending message")
	rate.sleep()