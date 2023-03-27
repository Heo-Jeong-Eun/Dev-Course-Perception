#include "ros/ros.h"
#include <iostream>
#include "msg_send_cpp/my_msg_cpp.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "msg_sender", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<msg_send_cpp::my_msg_cpp>("msg_to_xycar", 1);
    msg_send_cpp::my_msg_cpp msg;

    msg.first_name = "gildong";
    msg.last_name = "Hong";
    msg.id_number = 20041003;
    msg.phone_number = "010-8990-3003";

    ros::Rate rate(1);
    while (ros::ok()) {
        pub.publish(msg);
        std::cout<<"sending message"<<std::endl;
        rate.sleep();

    }
}