## **RVIZ 모터와 센서 Viewer 통합**

- **RVIZ 가상 공간에서 8자 주행을 하는 자이카 모델에 라이다 센서와 IMU 센서 뷰어를 통합한다.**

- **3D 모델링 된 차량이 8자 주행을 하면서 주변 장애물까지 거리값을 Range로 표시하고, IMU 센싱값에 따라 차체가 기울어진다.**


- **Python File**
    - **odom_imu.py** 

        ```python
        #!/usr/bin/env python

        # JointState, Imu 토픽 사용
        import math
        from math import sin, cos, pi
        import rospy
        import tf
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
        from sensor_msgs.msg import JointState
        from sensor_msgs.msg import Imu

        global Angle
        global Imudata

        # JointState 토픽 콜백 함수 
        def callback(msg):
            global Angle
            Angle = msg.position[msg.name.index("front_left_hinge_joint")]

        rospy.Subscriber('joint_states', JointState, callback)

        def callback_imu(msg):
            global Imudata
            Imudata[0] = msg.orientation.x
            Imudata[1] = msg.orientation.y
            Imudata[2] = msg.orientation.z
            Imudata[3] = msg.orientation.w

        rospy.Subscriber('imu', Imu, callback_imu)

        rospy.init_node('odometry_publisher')

        Imudata = tf.transformations.quaternion_from_euler(0, 0, 0)

        odom_pub = rospy.Publisher("odom", Odometry, queue_size = 50)
        odom_broadcaster = tf.TransformBroadcaster()

        current_time = rospy.Time.now()
        last_time = rospy.Time.now()

        r = rospy.Rate(30.0)

        current_speed = 0.4
        wheel_base = 0.2
        x_ = 0
        y_ = 0
        yaw_ = 0
        Angle = 0

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            dt = (current_time - last_time).to_sec()

            current_steering_angle = Angle
            current_angular_velocity = current_speed * math.tan(current_steering_angle) / wheel_base

            x_dot = current_speed * cos(yaw_)
            y_dot = current_speed * sin(yaw_)
            x_ += x_dot * dt
            y_ += y_dot * dt
            yaw_ += current_angular_velocity * dt		
            
            # IMU 값을 Odometry에 담아 전달한다.
            odom_quat = Imudata

            # odometry 정보를 차체에 전달하는 base_link에 연결
            # IMU 값에 따라 차제가 움직이게 하기 위함이다. 
            odom_broadcaster.sendTransform(
                (x_, y_, 0.),
                odom_quat,
                current_time,
                "base_link",
                "odom"
            )

            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = "odom"

            # pose 설정, position, orientation
            odom.pose.pose = Pose(Point(x_, y_, 0.), Quaternion(*odom_quat))

            # frame_id 설정
            odom.child_frame_id = "base_link"

            # 메시지 publish
            odom_pub.publish(odom)

        last_time = current_time
            r.sleep()
        ```


- **Launch File**
    - **rviz_all.launch**

        ```xml
        <launch>
            <param name="robot_description" textfile="$(find rviz_all)/urdf/rviz_all.urdf" />
            <param name="use_gui" value="true" />

            <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true" 
                    args="-d $(find rviz_all)/rviz/rviz_all.rviz"/>
            
            <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
        
            <node name="driver" pkg="rviz_xycar" type="odom_8_drive.py" /> 
            <node name="odometry" pkg="rviz_all" type="odom_imu.py" />
            <node name="motor" pkg="rviz_xycar" type="converter.py" />

            <node name="rosbag_play" pkg="rosbag" type="play" output="screen" required="true" args="$(find rviz_lidar)/src/lidar_topic.bag" />
            <node name="lidar" pkg="rviz_lidar" type="lidar_urdf.py" output="screen" />
            
            <node name="imu" pkg="rviz_imu" type="imu_generator.py" />
        </launch>
        ```


- **URDF File** 
    - **rviz_all.urdf** 

        ```xml
        <?xml version="1.0" ?>

        <robot name="xycar" xmlns:xacro="http://www.ros.org/wiki/xacro">
            
            <link name="base_link" />
            <link name="baseplate">
                <visual>
                    <material name="acrylic" />
                    <origin rpy="0 0 0" xyz="0 0 0" />
                    <geometry>
                        <box size="0.5 0.2 0.07" />
                    </geometry>
                </visual>
            </link>
                    
            <joint name="base_link_to_baseplate" type="fixed">
                <parent link="base_link" />
                <child link="baseplate" />
                <origin rpy="0 0 0" xyz="0 0 0" />
            </joint>

            <link name="front_mount">
                <visual>
                    <material name="blue" />
                    <origin rpy="0 0.0 0" xyz="-0.105 0 0" />
                    <geometry>
                        <box size="0.50 0.12 0.01" />
                    </geometry>
                </visual>
            </link>

            <joint name="baseplate_to_front_mount" type="fixed">
                <parent link="baseplate" />
                <child link="front_mount" />
                <origin rpy="0 0 0" xyz="0.105 0 -0.059" />
            </joint>

            <link name="front" />
            <joint name="baseplate_to_front" type="fixed">
                <parent link="baseplate" />
                <child link="front" />
                <origin rpy="0 0 0" xyz="0.25 0 0" />
            </joint>
            
            <link name="back" />
            <joint name="baseplate_to_back" type="fixed">
                <parent link="baseplate" />
                <child link="back" />
                <origin rpy="0 0 3.14" xyz="-0.25 0 0" />
            </joint>
            
            <link name="left" />
            <joint name="baseplate_to_left" type="fixed">
                <parent link="baseplate" />
                <child link="left" />
                <origin rpy="0 0 1.57" xyz="0 0.1 0" />
            </joint>
            
            <link name="right" />
            <joint name="baseplate_to_right" type="fixed">
                <parent link="baseplate" />
                <child link="right" />
                <origin rpy="0 0 -1.57" xyz="0 -0.1 0" />
            </joint>

            <link name="front_shaft">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.285" radius="0.018" />
                    </geometry>
                </visual>
            </link>

            <joint name="front_mount_to_front_shaft" type="fixed">
                <parent link="front_mount" />
                <child link="front_shaft" />
                <origin rpy="0 0 0" xyz="0.105 0 -0.059" />
            </joint>

            <link name="rear_shaft">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.285" radius="0.018" />
                    </geometry>
                </visual>
            </link>

            <joint name="rear_mount_to_rear_shaft" type="fixed">
                <parent link="front_mount" />
                <child link="rear_shaft" />
                <origin rpy="0 0 0" xyz="-0.305 0 -0.059" />
            </joint>

            <link name="front_right_hinge">
                <visual>
                    <material name="white" />
                    <origin rpy="0 0 0" xyz="0 0 0" />
                    <geometry>
                        <sphere radius="0.015" />
                    </geometry>
                </visual>
            </link>

            <joint name="front_right_hinge_joint" type="revolute">
                <parent link="front_shaft" />
                <child link="front_right_hinge" />
                <origin rpy="0 0 0" xyz="0 -0.1425 0" />
                <axis xyz="0 0 1" />
                <limit effort="10" lower="-0.34" upper="0.34" velocity="100" />
            </joint>

            <link name="front_left_hinge">
                <visual>
                    <material name="white" />
                    <origin rpy="0 0 0" xyz="0 0 0" />
                    <geometry>
                        <sphere radius="0.015" />
                    </geometry>
                </visual>
            </link>

            <joint name="front_left_hinge_joint" type="revolute">
                <parent link="front_shaft" />
                <child link="front_left_hinge" />
                <origin rpy="0 0 0" xyz="0 0.14 0" />
                <axis xyz="0 0 1" />
                <limit effort="10" lower="-0.34" upper="0.34" velocity="100" />
            </joint>

            <link name="front_right_wheel">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.064" radius="0.07" />
                    </geometry>
                </visual>
            </link>

            <joint name="front_right_wheel_joint" type="continuous">
                <parent link="front_right_hinge" />
                <child link="front_right_wheel" />
                <origin rpy="0 0 0" xyz="0 0 0" />
                <axis xyz="0 1 0" />
                <limit effort="10" velocity="100" />
            </joint>
            
            <link name="front_left_wheel">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.064" radius="0.07" />
                    </geometry>
                </visual>
            </link>

            <joint name="front_left_wheel_joint" type="continuous">
                <parent link="front_left_hinge" />
                <child link="front_left_wheel" />
                <origin rpy="0 0 0" xyz="0 0 0" />
                <axis xyz="0 1 0" />
                <limit effort="10" velocity="100" />
            </joint>

            <link name="rear_right_wheel">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.064" radius="0.07" />
                    </geometry>
                </visual>
            </link>

            <joint name="rear_right_wheel_joint" type="continuous">
                <parent link="rear_shaft" />
                <child link="rear_right_wheel" />
                <origin rpy="0 0 0" xyz="0 -0.14 0" />
                <axis xyz="0 1 0" />
                <limit effort="10" velocity="100" />
            </joint>

            <link name="rear_left_wheel">
                <visual>
                    <material name="black" />
                    <origin rpy="1.57 0 0" xyz="0 0 0" />
                    <geometry>
                        <cylinder length="0.064" radius="0.07" />
                    </geometry>
                </visual>
            </link>

            <joint name="rear_left_wheel_joint" type="continuous">
                <parent link="rear_shaft" />
                <child link="rear_left_wheel" />
                <origin rpy="0 0 0" xyz="0 0.14 0" />
                <axis xyz="0 1 0" />
                <limit effort="10" velocity="100" />
            </joint>

            <material name="black">
                <color rgba="0.0 0.0 0.0 1.0" />
            </material>
            
            <material name="blue">
                <color rgba="0.0 0.0 0.8 1.0" />
            </material>
            
            <material name="green">
                <color rgba="0.0 0.8 0.0 1.0" />
            </material>
            
            <material name="red">
                <color rgba="0.8 0.0 0.0 1.0" />
            </material>
            
            <material name="white">
                <color rgba="1.0 1.0 1.0 1.0" />
            </material>

            <material name="orange">
                <color rgba="1.0 0.423529411765 0.0392156862745 1.0" />
            </material>

            <material name="brown">
                <color rgba="0.870588235294 0.764705882353 1.0" />
            </material>

            <material name="acrylic">
                <color rgba="1.0 1.0 1.0 0.4" />
            </material>
        </robot>
        ```

- **Code Run**

    ```shell
    $ roslaunch rviz_all rviz_all.launch 
    $ rqt_graph
    ```

    <img src = 'img/RVIZ Motor + Sensor Viewer-Graph Debug.png' alt = 'RVIZ Motor + Sensor Viewer-Graph Debug.png' width='500' height='300'>