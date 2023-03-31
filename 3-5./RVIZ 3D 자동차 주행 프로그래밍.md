- ## **RVIZ 3D 자동차 주행 프로그래밍**

- **RVIZ 가상 공간에 있는 3D 자동차를 주행시킨다.**

- 동작 과정
    1. 8자 주행 프로그램이 모터 제어 메시지를 보낸다. → /xycar_motor 토픽

    2. 변환 프로그램이 받아 변환, /joint_states 토픽을 만들어 발행한다. 

    3. 오도메트리 프로그램이 받아 변환, /odom 토픽을 만들어 발행한다.


- **자이카 모터 제어 토픽**
    - **xycar_motor/xycar_motor**
    
        ```shell
        $ rosmsg show xycar_motor/xycar_motor
        $ rostopic echo /xycar_motor
        ```

        <img src = 'img/RVIZ 3D xycar_motor, xycar_motor.png' alt = 'RVIZ 3D xycar_motor/xycar_motor.png' width='500' height='100'>


- **joint 상태 정보 토픽**
    - **sensor_msgs/JointState**
        
        ```shell
        $ rosmsg show sensor_msgs/JointState
        ```

        <img src = 'img/RVIZ 3D joint.png' alt = 'RVIZ 3D joint' width='500' height='100'>



- **odom 토픽**
    - **nav_msgs/Odometry**
        
        ```shell
        $ rosmsg show nav_msgs/Odometry
        ```

        <img src = 'img/RVIZ 3D odom.png' alt = 'RVIZ 3D odom' width='500' height='300'>


- **Python File**
    - **odom_8_drive.py** 
        - 8자 주행을 하도록 운전하는 프로그램
            
            ```python
            #!/usr/bin/env python
            
            # import 필요한 모듈, 모터 제어를 위한 토픽 메시지 타입을 가져온다.
            import rospy
            import time
            from xycar_motor.msg import xycar_motor
            
            motor_control = xycar_motor()
            
            # driver 이름으로 노드 만들기
            rospy.init_node('driver')
            
            # xycar_motor 토픽의 발행 준비
            pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
            
            # 토픽을 발행하는 함수를 만든다. 
            # angle 값과 speed 값을 인자로 받고 토픽에 담아 발행
            def motor_pub(angle, speed):
                global pub
                global motor_control
            
                motor_control.angle = angle
                motor_control.speed = speed
            
                pub.publish(motor_control)
            
            # 차량의 속도 고정
            # 구동 속도를 3으로 설정
            # 차량의 조향각을 바꿔가며 8자로 주행한다.
            # 핸들을 꺾어 8자 모양으로 주행한다. (좌 + 직진 + 우 + 직진)
            speed = 3
            
            while not rospy.is_shutdown():
                # 핸들 -> 최대로 왼쪽으로 꺾어 주행, (좌회전)
                angle = -50 
                for i in range(40):
                    motor_pub(angle, speed)
                    time.sleep(0.1)
            
                # 핸들 -> 중앙으로 놓고 주행, (직진)
                angle = 0
                for i in range(30):
                    motor_pub(angle, speed)
                    time.sleep(0.1)
            
                # 핸들 -> 최대로 오른쪽으로 꺾어 주행, (우회전)
                angle = 50
                for i in range(40):
                    motor_pub(angle, speed)
                    time.sleep(0.1)
            
                # 핸들 -> 중앙으로 놓고 주행, (직진)
                angle = 0
                for i in range(30):
                    motor_pub(angle, speed)
                    time.sleep(0.1)
            ```
    
    - **rviz_odom.py**
        - converter 노드가 보내는 /joint_states 토픽을 받아 바퀴의 방향과 회전속도 정보를 획득
        
        - 그 정보를 바탕으로 오도메트리 데이터를 만들어 /odom 토픽에 담아 발행한다.
            
            ```python
            #!/usr/bin/env python
            
            import math
            from math import sin, cos, pi
            import rospy
            import tf
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
            from sensor_msgs.msg import JointState
            
            global Angle
            
            def callback(msg):
                global Angle
                Angle = msg.position[msg.name.index("front_left_hinge_joint")]
            
            rospy.Subscriber('joint_states', JointState, callback)
            rospy.init_node('odometry_publisher')
            
            odom_pub = rospy.Publisher("odom", Odometry, queue_size=50)
            odom_broadcaster = tf.TransformBroadcaster()
            
            current_time = rospy.Time.now()
            last_time = rospy.Time.now()
            
            # 1초에 30번 수행 
            r = rospy.Rate(30.0)
            
            # 속도 = 초속 40cm
            current_speed = 0.4
            # 축간 거리는 20cm, 앞 바퀴와 뒷 바퀴 중심간 거리
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
                
                odom_quat = tf.transformations.quaternion_from_euler(0, 0, yaw_)
            
                # 위치 정보자에 대한 발행을 준비한다. 
                # odom과 base_link를 연결하는 효과
                odom_broadcaster.sendTransform(
                    (x_, y_, 0.),
                    odom_quat,
                    current_time,
                    "base_link",
                    "odom"
            )
            
                # Odometry 메시지의 헤더 만들기
                odom = Odometry()
                odom.header.stamp = current_time
            odom.header.frame_id = "odom"
            
                # position 값 채우기 
            odom.pose.pose = Pose(Point(x_, y_, 0.), Quaternion(*odom_quat))
            
                # 속도값 채우기
            odom.child_frame_id = "base_link"
            # odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, yaw_))
            
                # /odom 토픽 발행하기
            odom_pub.publish(odom)
            
            last_time = current_time
            r.sleep()
            ```


- **RVIZ 뷰어 설정 파일**
    - **rviz_odom.rviz**
        - RVIZ 뷰어 화면에 차량 3D 그림이 잘 표시되도록 설정
        - RVIZ 종료 시 설정 내용을 ‘Save’하면 된다.
        - 저장 위치는 .launch 파일에서 지정할 수 있다.


- **Launch File**
    - **rviz_odom.launch**
        
        ```xml
        <launch>
        	<param name="robot_description" textfile="$(find rviz_xycar)/urdf/xycar_3d.urdf"/>
        	<param name="use_gui" value="true"/>
        
        	<!-- rviz display -->
        	<node name="rviz_visualizer" pkg="rviz" type="rviz" required="true" 
        			args="-d $(find rviz_xycar)/rviz/rviz_odom.rviz"/>
        	
        	<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>
          
        	<node name="driver" pkg="rviz_xycar" type="rviz_8_drive.py" /> 
        	<node name="converter" pkg="rviz_xycar" type="converter.py" />
        	<node name="odometry" pkg="rviz_xycar" type="rviz_odom.py" />
        </launch>
        ```


- **Code Run**

    ```shell
    $ roslaunch rviz_xycar rviz_odom.launch
    $ rqt_graph
    ```

    <img src = 'img/RVIZ 3D Launch File Run.png' alt = 'RVIZ 3D Launch File Run' width='500' height='300'>

    <img src = 'img/RVIZ 3D Graph Run.png' alt = 'RVIZ 3D Graph Run' width='500' height='250'>