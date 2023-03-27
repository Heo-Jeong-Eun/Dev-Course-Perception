#include "ros/ros.h"
#include <cmath>
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "tf/transform_broadcaster.h"
#include "tf/transform_datatypes.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "odometry_publisher", ros::init_options::AnonymousName);
    ros::NodeHandle nh;
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);

    tf::TransformBroadcaster odom_broadcaster;
    tf::Quaternion odom_quat;
    tf::Transform transform;
    nav_msgs::Odometry odom;
    geometry_msgs::Point point;
    geometry_msgs::Quaternion quat;
    geometry_msgs::Pose pose;
    geometry_msgs::Twist twist;

    double x = 0.0, y = 0.0, th = 0.0;
    double vx = 0.1, vy = -0.1, vth = 0.1;
    double dt, delta_x, delta_y, delta_th;

    ros::Time current_time = ros::Time::now();
    ros::Time last_time = ros::Time::now();

    ros::Rate rate(60);
    while (ros::ok()) {
        current_time = ros::Time::now();

        dt = (current_time - last_time).toSec();
        delta_x = (vx * cos(th) - (vy * sin(th))) * dt;
        delta_y = (vx * sin(th) + (vy * cos(th))) * dt;
        delta_th = vth * dt;

        x += delta_x;
        y += delta_y;
        th += delta_th;

        odom_quat.setRPY(0, 0, th);
        odom_quat = odom_quat.normalize();

        transform.setOrigin(tf::Vector3(x, y, 0));
        transform.setRotation(odom_quat);
        odom_broadcaster.sendTransform(
            tf::StampedTransform(
                transform,
                current_time,
                "base_link",
                "odom"
            )
        );

        odom.header.stamp = current_time;
        odom.header.frame_id = "odom";

        
        point.x = x; point.y = y; point.z = 0;
        tf::quaternionTFToMsg(odom_quat, quat);
        
        pose.position = point; pose.orientation = quat;
        odom.pose.pose = pose;

        odom.child_frame_id = "base_link";

        twist.linear.x = vx; twist.linear.y = vy; twist.linear.z = 0;
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = vth;
        odom.twist.twist = twist;

        odom_pub.publish(odom);

        last_time = current_time;
        rate.sleep();

    }
}