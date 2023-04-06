- ## **RVIZ 기반 IMU Viewer**

- **RVIZ에서 IMU 데이터 제작하기**


- **imu_generator.py** 

    ```python
    #!/usr/bin/env python

    # quaternion_from_euler 함수 사용을 위해 import 
    import rospy, math, os, rospkg
    from sensor_msgs.msg import Imu

    from tf.transformations import quaternion_from_euler

    # 라디안 / 각도 변환 계산식
    degree2rad = float(math.pi) / float(180.0) 
    rad2degrees = float(180.0) / float(math.pi)

    # 노드 이름 지정, 생성
    rospy.init_node("imu_generator")

    # Imu 타입의 데이터를 담아 /imu 이름의 토픽으로 발행할 것을 정의 
    pub = rospy.Publisher('imu', Imu, queue_size=1)

    data = []

    # imu.data.txt 파일을 찾아 한줄씩 읽는다. 
    path = rospkg.RosPack().get_path('rviz_imu')+"/src/imu_data.txt"
    f = file(path, "r")
    lines = f.readlines()

    # roll, pitch, yaw 숫자값만 추출한다. 
    for line in lines:
        temp = line.split(",")
        extract = []
        for i in temp:
            extract.append(float(i.split(":")[1]))
        data.append(extract)

    # Imu 메시지를 만들과 frame_id 값으로 map을 넣는다. 
    imuMsg = Imu()
    imuMsg.header.frame_id = 'map'

    r = rospy.Rate(10)
    seq = 0

    # imu.data.txt 파일의 line 수만큼 반복 
    for j in range(len(data)):
        msg_data = quaternion_from_euler(data[j][0], data[j][1], data[j][2])

        # 오일러 데이터(roll, pitch, yaw)를 쿼터니언(x, y, z, w)로 변환
        imuMsg.orientation.x = msg_data[0]
        imuMsg.orientation.y = msg_data[1]
        imuMsg.orientation.z = msg_data[2]
        imuMsg.orientation.w = msg_data[3]

        # Imu 메시지 헤더 채우기 -> stamp에 시간 정보, seq에 순차 number
        imuMsg.header.stamp = rospy.Time.now()
        imuMsg.header.seq = seq
        seq += 1

        # /imu 토픽 발행, Rate(10) == 1초에 10번씩 
        pub.publish(imuMsg)
        r.sleep()
    ```


- **imu_generator.launch**

    ```xml
    <launch>
        <node name="rviz_visualizer" pkg="rviz" type="rviz" required="true" args="-d $(find rviz_imu)/rviz/imu_generator" />
        <node pkg="imu_generator" pkg="rviz_imu" type="imu_generator.py" />
    </launch>
    ```


- **imu_generator.rivz**

    - RVIZ 실행 → imu 설정 → imu_generator.rivz로 저장


- **Code Run**

    ```shell
    $ roslaunch rivz_imu imu_generator.launch
    $ rostopic echo /imu
    $ rqt_graph
    ```

    <img src = 'img/RVIZ IMU Viewer-RVIZ.png' alt = 'RVIZ IMU Viewer-RVIZ' width='500' height='300'>


    <img src = 'img/RVIZ IMU Viewer-Topic.png' alt = 'RVIZ IMU Viewer-Topic' width='500' height='400'>


    <img src = 'img/RVIZ IMU Viewer-Graph.png' alt = 'RVIZ IMU Viewer-Graph' width='500' height='300'> 