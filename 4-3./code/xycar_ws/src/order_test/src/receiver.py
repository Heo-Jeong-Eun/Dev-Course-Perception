#!/usr/bin/env python

import rospy
from std_msgs.msg import String

name = "receiver"
pub_topic = "start_ctl"
sub_topic = "msg_to_receiver"

def callback(data):
	rospy.loginfo("I heard %s", data.data)

rospy.init_node(name)
rospy.Subscriber(sub_topic, String, callback)
pub = rospy.Publisher(pub_topic, String, queue_size=1)

rate = rospy.Rate(10)
hello_str = String()

rospy.sleep(1)

sq = ["first", "second", "third", "fourth"]
pub_msg = String()

for i in sq:
	pub_msg.data = i + ":go"
	pub.publish(pub_msg)
	rospy.sleep(3)

rospy.spin()
