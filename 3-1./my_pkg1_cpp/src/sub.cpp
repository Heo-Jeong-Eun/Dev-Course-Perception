#include "ros/ros.h"
#include "turtlesim/Pose.h"
#include <iostream>

void callback(turtlesim::Pose msg) {
    std::cout<<msg.x<<", "<<msg.y;
}

int main(int argc, char **argv) {
    ros::NodeHandle nh;
    ros::init(argc, argv, "my_listener", ros::init_options::AnonymousName);
    ros::Subscriber sub = nh.subscribe<turtlesim::Pose>("/turtle1/pose", 1, callback);
    ros::spin();
    
    return 0;
}