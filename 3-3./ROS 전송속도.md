- ## **데이터 크기에 따른 전송속도는 어떻게 되는가 ?**

- **sender_speed.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "sender"
    pub_topic = "long_string"

    rospy.init_node(name, anonymous=True)
    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    hello_str = String()
    rate = rospy.Rate(1)

    pub_size = 1000000 # 1M byte
    # pub_size = 5000000 # 5M byte
    # pub_size = 10000000 # 10M byte
    # pub_size = 20000000 # 20M byte
    # pub_size = 50000000 # 50M byte

    my_string = ""

    for i in range(pub_size):
        my_string += "#"

    while not rospy.is_shutdown():
        hello_str.data = my_string + ":" + str(rospy.get_time())
        pub.publish(hello_str)
        # rospy.loginfo(str(hello_str.data).split(":")[1])
        rate.sleep()
    ```
            
- **receiver_speed.py**
            
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "receiver"
    sub_topic = "long_string"

    def callback(data):
        current_time = str(rospy.get_time())
        arrival_data = str(data.data).split(":")
        # 도착하는 토픽을 : 으로 자른다. 

        time_diff = float(current_time) - float(arrival_data[1])
        # 실제 토픽을 전송할 때 걸린 시간 
        string_size = len(arrival_data[0])
        # 문자열 사이즈를 저장해둔다. 
        rospy.loginfo(str(string_size) + " byte : " + str(time_diff) + "sceond")
        rospy.loginfo("speed : " + str(float(string_size) / time_diff) + "byte/s")

    rospy.init_node(name, anonymous=True)
    rospy.loginfo("Init")

    rospy.Subscriber(sub_topic, String, callback)
    rospy.spin()
    ```

- **정해진 크기의 데이터를 반복해 대량으로 전송한다.**
    - 보내는 쪽은 10분동안 시간을 정하고, 쉴 새 없이 보낸다.
    
    - 10분동안 몇 byte 보냈는지 체크해서 송신 속도를 계산한다.
    
    - 받는 쪽도 10분동안 시간을 정해놓고, 모두 얼마나 받았는지 체크해 수신 속도를 계산한다.
    
    - 단위는 300kbytes/sec
    
    - 어느정도 사이즈가 좋은지 결과를 분석해본다.

- **받는 쪽이 없는 경우는 어떻게 되는지 확인해본다.**
    - 토픽에 대해 구독하는 노드가 없으면 송신 속도가 더 빨라지는지 ? 아니면 상관 없는지 ?

- **sender_speed.py 코드**
    - Sender라는 이름으로 노드를 생성한다.
    
    - 발행하는 토픽 이름은 long_string, type = String
    
    - 1초에 한번씩 다양한 용량의 long_string을 발행한다. 이때 문자열은 #으로 가득 채운다.
    
    - 사이즈를 바꿔 1Mbyte, 5Mbyte, 10Mbyte, 20Mbyte, 50Mbyte를 전송한다.
    
    - 코드에 사이즈를 나열하고 안쓰는 사이즈는 주석처리하면 편리하다.

- **receiver_speed.py 코드**
    - Receiver라는 이름으로 노드를 생성한다.
    
    - 다양한 용량의 long_string을 수신, long_string 1개를 다 받으면 소요 시간을 화면에 출력한다.
    
    - 가능하면 속도도 출력한다. 단위 → Bps


- **현상 확인 및 분석**
    - 1Mbyte 전송 : 0.01초 소요, 전송 속도는 약 100MBps, 수행 시간이 거의 0으로 수렴해 나누기 에러가 발생한다. 
        
        <img src = 'img/ROS Speed 1Mbyte.png' alt = 'ROS Speed 1Mbyte' width='500' height='300'>
    
    - 5Mbyte 전송 : 0.01초 소요, 전송 속도는 약 500MBps, 5Mbyte 전송도 수행 시간이 0으로 수렴하는 경우가 나와 나누기 에러가 발생한다.
    
        <img src = 'img/ROS Speed 5Mbyte.png' alt = 'ROS Speed 5Mbyte' width='500' height='300'>
    
    - 10Mbyte 전송 : 0.03초 소요, 전송 속도는 약 333MBps, 속도가 많이 불규칙하다. 
        
        <img src = 'img/ROS Speed 10Mbyte.png' alt = 'ROS Speed 10Mbyte' width='500' height='300'>

    - 20Mbyte 전송 : 약 0.05초 소요, 전송 속도는 약 500MBps, 333MBps도 보인다. <br>
      **전체적으로 일정한 속도가 출력되고, 속도가 빠른 것은 20Mbyte를 전송했을 때라고 볼 수 있다.**
        
        <img src = 'img/ROS Speed 20Mbyte.png' alt = 'ROS Speed 20Mbyte' width='500' height='300'>

    - 50Mbyte 전송 : 약 0.1초 소요, 전송 속도는 약 270-350MBps, 전송 속도가 각각 다양하다. 
        
        <img src = 'img/ROS Speed 50Mbyte.png' alt = 'ROS Speed 50Mbyte' width='500' height='300'>