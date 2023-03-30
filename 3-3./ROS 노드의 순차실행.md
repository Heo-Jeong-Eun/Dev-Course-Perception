- ## **데이터 크기에 따른 전송속도는 어떻게 되는가 ?**

- **first.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "first"
    pub_topic = "msg_to_receiver"
    sub_topic = "start_ctl"

    OK = None

    def ctl_callback(data):
        global OK
        OK = str(data.data)

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, String, ctl_callback)

    while True:
        if OK == None:
            continue
        d = OK.split(":")
        if (len(d) == 2) and (d[0] == name) and (d[1] == "go"):
            break

    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    rate = rospy.Rate(2)
    hello_str = String()

    while not rospy.is_shutdown():
        hello_str.data = "my name is " + name
        pub.publish(hello_str)
        rate.sleep()
    ```

- **second.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "second"
    pub_topic = "msg_to_receiver"
    sub_topic = "start_ctl"

    OK = None

    def ctl_callback(data):
        global OK
        OK = str(data.data)

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, String, ctl_callback)

    while True:
        if OK == None:
            continue
        d = OK.split(":")
        if (len(d) == 2) and (d[0] == name) and (d[1] == "go"):
            break

    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    rate = rospy.Rate(2)
    hello_str = String()

    while not rospy.is_shutdown():
        hello_str.data = "my name is " + name
        pub.publish(hello_str)
        rate.sleep()
    ```

- **third.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "third"
    pub_topic = "msg_to_receiver"
    sub_topic = "start_ctl"

    OK = None

    def ctl_callback(data):
        global OK
        OK = str(data.data)

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, String, ctl_callback)

    while True:
        if OK == None:
            continue
        d = OK.split(":")
        if (len(d) == 2) and (d[0] == name) and (d[1] == "go"):
            break

    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    rate = rospy.Rate(2)
    hello_str = String()

    while not rospy.is_shutdown():
        hello_str.data = "my name is " + name
        pub.publish(hello_str)
        rate.sleep()
    ```

- **fourth.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "fourth"
    pub_topic = "msg_to_receiver"
    sub_topic = "start_ctl"

    OK = None

    def ctl_callback(data):
        global OK
        OK = str(data.data)

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, String, ctl_callback)

    while True:
        if OK == None:
            continue
        d = OK.split(":")
        if (len(d) == 2) and (d[0] == name) and (d[1] == "go"):
            break

    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    rate = rospy.Rate(2)
    hello_str = String()

    while not rospy.is_shutdown():
        hello_str.data = "my name is " + name
        pub.publish(hello_str)
        rate.sleep()
    ```
            
- **receiver.py**
            
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import String

    name = "receiver"
    pub_topic = "start_ctl"
    sub_topic = "msg_to_receiver"

    def callback(data):
        rospy.loginfo("I heard %s" data.data)

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, String, callback)
    pub = rospy.Publisher(pub_topic, String, queue_size=1)

    rate = rospy.Rate(10)
    hello_str = String()

    rospy.sleep()

    sq = ["first", "second", "third", "fourth"]
    pub_msg = String()

    for i in sq:
        pub_msg.data = i + ":go"
        pub.publish(pub_msg)
        rospy.sleep(3)

    rospy.spin()
    ```

- **sr_order.launch**

    ```xml
    <launch>
        <node name="receiver" pkg="order_test" type="receiver.py" output="screen" />
        <node name="first" pkg="order_test" type="first.py" output="screen" />
        <node name="second" pkg="order_test" type="second.py" output="screen" />
        <node name="third" pkg="order_test" type="third.py" output="screen" />
        <node name="fourth" pkg="order_test" type="fourth.py" output="screen" />
    </launch>
    ```


- **순서대로 receiver에 메시지를 보내도록 한다.**
    - receiver는 도착한 순서대로 출력한다. f → s → t → f

    - 앞에 있는 노드가 움직이기 전에 먼저 움직여서는 안된다. → 움직인다 == 토픽을 보내는 것으로 대신한다.

- **어떻게 동기를 맞추고 순서를 맞출 수 있을까 ?**
    - launch 파일을 이용해서 할 수 있을까 ?

    - ROS의 도움으로 할 수 있을까 ?

    - 아니면 프로그래밍을 해야하는 것인가 ?

- **Receiver.py 작성**
    - 토픽 이름은 msg_to_receiver, type = String

    - 적당한 시간 간격을 두고 my name is first부터 시작해서 my name is fourth까지 받아야 한다.

    - 테스트를 위해 받은 토픽이 순차적으로 오는지, 화면에 출력하도록 한다.

- **Frist, Second, Third, Fourth.py 작성**
    - 자기 이름에 맞춰 Third 노드인 경우 my name is third라는 msg_to_receiver 토픽을 receiver로 전송한다.

    - 자기보다 서수가 앞선 노드를 먼저 보내서는 안된다.
    
- **패키지는 order_test라는 이름을 만들고, launch 디렉토리 생성 후 cm → catkin_create_pkg order_test std_msgs rospy**


- **현상 확인 및 분석**
    - 협업 해야하는 노드를 순서대로 기동시킬 수 있는지 확인해야 한다.

    - launch 파일로 실행하게 되면 노드를 순서대로 구동시킬 수 없다. → 랜덤

        <img src = 'img/ROS Sequential Execution (Problem Situation).png' alt = 'ROS Sequential Execution (Problem Situation)' width='500' height='300'>


- **해결책 적용 결과**
    - **Receiver가 보내라는 사인을 주기 전까지, 각 서수 노드들을 기다리게 하는 방법으로 구현한다.** 

        <img src = 'img/ROS Sequential Execution (Troubleshooting) 1.png' alt = 'ROS Sequential Execution (Troubleshooting) 1' width='500' height='100'>

        <img src = 'img/ROS Sequential Execution (Troubleshooting) 2.png' alt = 'ROS Sequential Execution (Troubleshooting) 2' width='500' height='300'>

        <img src = 'img/ROS Sequential Execution rqt_graph.png' alt = 'ROS Sequential Execution rqt_graph' width='500' height='300'>
