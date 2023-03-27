#include <iostream>
#include "ros/ros.h"
#include "msg_send_cpp/my_msg_cpp.h"

void callback(msg_send_cpp::my_msg_cpp msg) {
    std::cout<<"1. Name : "<<msg.last_name<<msg.first_name<<std::endl;
    std::cout<<"2. ID : "<<msg.id_number<<std::endl;
    std::cout<<"3. Phone Number : "<<msg.phone_number<<std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "msg_receiver", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe<msg_send_cpp::my_msg_cpp>("msg_to_xycar", 1, callback);
    ros::spin();
}