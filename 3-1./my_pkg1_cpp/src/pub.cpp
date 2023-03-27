#include "ros/ros.h"
#include "geometry_msgs/Twist.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_node", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel", 1);

    geometry_msgs::Twist msg;
    msg.linear.x = 2.0;
    msg.linear.y = 2.0;
    msg.linear.z = 2.0;

    msg.angular.x = 0.0;
    msg.angular.y = 0.0;
    msg.angular.z = 1.8;

    ros::Rate rate(1);

    while (ros::ok())
    {
        pub.publish(msg);
        rate.sleep();
    }
    return 0;
}