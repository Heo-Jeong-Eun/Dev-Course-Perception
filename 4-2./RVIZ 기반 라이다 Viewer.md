- ## **RVIZ 기반 라이다 Viewer**

- **RVIZ에서 라이다 정보를 Range로 표시**


- **데이터 전달 흐름**
    - lidar_topic.bag 파일에 저장된 라이다 토픽을 ROSBAG으로 하나씩 발행

    - scan 토픽에서 장애물의 거리정보를 꺼내 scan1, 2, 3, 4 4개의 토픽에 담아 발행한다.

    - RVIZ에서는 Range 형식으로 거리 정보를 시각화해서 보여준다.


- **Python File**
    - **lidar_urdf.py** 

        ```python
        #!/usr/bin/env python

        # Range, Header import
        import serial, time, rospy
        from sensor_msgs.msg import LaserScan
        from sensor_msgs.msg import Range
        from std_msgs.msg import Header

        lidar_points = None

        # 라이다 토픽이 도착하면 실행되는 콜백 함수, 거리 정보를 lidar_points에 옮겨 저장한다. 
        def lidar_callback(data):
            global lidar_points
            lidar_points = data.ranges

        rospy.init_node('lidar')
        rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size = 1)

        # 4개의 토픽 발행 준비
        pub1 = rospy.Publisher('scan1', Range, queue_size = 1)
        pub2 = rospy.Publisher('scan2', Range, queue_size = 1)
        pub3 = rospy.Publisher('scan3', Range, queue_size = 1)
        pub4 = rospy.Publisher('scan4', Range, queue_size = 1)

        # Range, Header 정보 채우기, 원뿔 모양의 Range 표시에 필요한 정보를 채운다. 
        msg = Range()
        h = Header()

        msg.radiation_type = Range().ULTRASOUND
        msg.min_range = 0.02
        msg.max_range = 2.0
        msg.field_of_view = (30.0/180.0) * 3.14

        while not rospy.is_shutdown():	
            if lidar_points == None:
                continue
            
            h.frame_id = "front"
            msg.header = h
            msg.range = lidar_points[90]
            pub1.publish(msg)

            h.frame_id = "right"
            msg.header = h
            msg.range = lidar_points[180]
            pub2.publish(msg)

            h.frame_id = "back"
            msg.header = h
            msg.range = lidar_points[270]
            pub3.publish(msg)

            h.frame_id = "left"
            msg.header = h
            msg.range = lidar_points[0]
            pub4.publish(msg)
            
            time.sleep(0.5)
        ```


- **Launch File**
    - **lidar_urdf.launch**

        ```xml
        <launch>
            <param name="robot_description" textfile="$(find rviz_lidar)/urdf/lidar_urdf.urdf" />
            <param name="use_gui" value="true" />
            
            <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true" args="-d $(find rviz_lidar)/rviz/lidar_urdf.rviz" />
            <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" />
            
            <node name="rosbag_play" pkg="rosbag" type="play" output="screen" required="true" args="$(find rviz_lidar)/src/lidar_topic.bag" />
            <node name="lidar" pkg="rviz_lidar" type="lidar_urdf.py" output="screen" />
        </launch>
        ```


- **URDF File** 
    - **lidar_urdf.urdf**

        ```xml
        <?xml version="1.0" ?>
        
        <robot name="xycar" xmlns:xacro="http://www.ros.org/wiki/xacro">
            
            <!-- link를 2개 만들고, joint로 연결 -->

            <!-- link base_link -->
            <link name="base_link" />

                <!-- link baseplate -->
                <link name="baseplate">
                    <visual>
                        <material name="red" />
                        <origin rpy="0 0 0" xyz="0 0 0" />
                        <geometry>
                            <box size="0.2 0.2 0.07" />
                        </geometry>
                    </visual>
                </link>
                    
            <!-- joint base_link_to_baseplate -->
            <joint name="base_link_to_baseplate" type="fixed">
                <parent link="base_link" />
                <child link="baseplate" />
                <origin rpy="0 0 0" xyz="0 0 0" />
            </joint>
            
            <!-- front -->
            <link name="front" />
            
            <!-- joint baseplate_to_front -->
            <!-- rpy 모두 0이므로 회전 X, 중심에서 x축으로 0.1m 전진해서 위치 -->
            <joint name="baseplate_to_front" type="fixed">
                <parent link="baseplate" />
                <child link="front" />
                <origin rpy="0 0 0" xyz="0.1 0 0" />
            </joint>
            
            <!-- back -->
            <link name="back" />
            
            <!-- joint baseplate_to_back -->
            <!-- y 3.15 pi이므로 180도 회전, 중심에서 -x축으로 0.1m 전진해서 위치 -->
            <joint name="baseplate_to_back" type="fixed">
                <parent link="baseplate" />
                <child link="back" />
                <origin rpy="0 0 3.14" xyz="-0.1 0 0" />
            </joint>
            
            <!-- left -->
            <link name="left" />
            
            <!-- joint baseplate_to_left -->
            <!-- y 1.57 절반 pi이므로 90도 회전, 중심에서 y축으로 0.1m 전진해서 위치 -->
            <joint name="baseplate_to_left" type="fixed">
                <parent link="baseplate" />
                <child link="left" />
                <origin rpy="0 0 1.57" xyz="0 0.1 0" />
            </joint>
            
            <!-- right -->
            <link name="right" />
            
            <!-- joint baseplate_to_right -->
            <!-- y -1.57 절반 pi이므로 -90도 회전, 중심에서 -y축으로 0.1m 전진해서 위치 -->
            <joint name="baseplate_to_right" type="fixed">
                <parent link="baseplate" />
                <child link="right" />
                <origin rpy="0 0 -1.57" xyz="0 -0.1 0" />
            </joint>
            
            <!-- color part -->
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
        </robot>
        ```
    
    - world 중앙에 RED 박스를 만들고 4 방향에 센서 프레임을 연결한다.
    
    - base_link에 가로 세로 20cm RED 박스 baseplate를 만들어 연결한다.

    - 센서는 x, y 축을 기준으로 중심에서 10cm씩 이동시켜 박스의 끝 부분에 배치한다.


- **Code Run**

    ```shell
    $ roslaunch rviz_lidar lidar_urdf.launch
    $ rqt_graph
    ```

    <img src = 'img/RVIZ Lidar-RVIZ.png' alt = 'RVIZ Lidar-RVIZ' width='500' height='300'>

    <img src = 'img/RVIZ Lidar-Graph Debug.png' alt = 'RVIZ Lidar-Graph Debug' width='500' height='300'>

    <img src = 'img/RVIZ Lidar-Graph.png' alt = 'RVIZ Lidar-Graph' width='500' height='300'>

     ```shell
    $ rostopic echo /scan
    ```

    <img src = 'img/RVIZ Lidar-Topic.png' alt = 'RVIZ Lidar-Topic' width='500' height='300'>