# 실습 터틀심 8자 주행 

시간차를 두고 번갈아가며 각속도를 역회전 시킨다. 

```py
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

rospy.init_node('my_node', anonymous=True) 
pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10) 
rate = rospy.Rate(1)
msg = Twist()

while not rospy.is_shutdown():
	for i in range(0, 4):
		msg.linear.x = 3.0
		msg.linear.y = 0.0
		msg.linear.z = 0.0
		msg.angular.x = 0.0
		msg.angular.y = 0.0
		msg.angular.z = -3.0
		pub.publish(msg)
    		rate.sleep()	
	for j in range(0, 4):
		msg.linear.x = 3.0
		msg.linear.y = 0.0
		msg.linear.z = 0.0
		msg.angular.x = 0.0
		msg.angular.y = 0.0
		msg.angular.z = 3.0
		pub.publish(msg)
    		rate.sleep()

while not rospy.is_shutdown():
	pub.publish(msg)
    	rate.sleep()
```