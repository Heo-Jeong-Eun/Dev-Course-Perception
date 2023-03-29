- ## **도착하는 데이터를 처리하지 못하면 어떻게 되는가 ?**

- **sender_overflow.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import Int32

    name = "sender"
    pub_topic = "my_topic"

    rospy.init_node(name)
    pub = rospy.Publisher(pub_topic, Int32, queue_size=1)

    rate = rospy.Rate(1000)
    # 1초당 1000번
    count = 1

    while (pub.get_num_connections() == 0):
        continue

    while not rospy.is_shutdown():
        pub.publish(count)
        count += 1
        rate.sleep()
    ```
            
- **receiver_overflow.py**
            
    ```python
    #!/usr/bin/env python

    import rospy
    from std_msgs.msg import Int32

    name = "receiver"
    sub_topic = "my_topic"

    def callback(msg):
        rospy.loginfo("callback is being processed")
        rospy.sleep(5)
        # overhead가 필요하므로 5초동안 일을 하지 않고 쉬도록 만든다. 
        print msg.data

    rospy.init_node(name)
    rospy.Subscriber(sub_topic, Int32, callback, queue_size=1)
    rospy.spin()
    ```

- **받는 쪽이 버벅 되도록 만들고 데이터를 대량으로 전송한다.**
    - 구독자의 콜백 함수 안에 시간이 많이 걸리는 코드를 넣어서 토픽 처리에 시간이 걸리도록 만든다.

- **콜백 함수가 끝나지 않았는데 토픽이 새로 도착하면 어떻게 되는가 ?**
    - 도착한 토픽은 임시에 쌓이는가 ? 이후에 꺼내 처리할 수 있는가 ?

    - 아니면 그냥 없어지는가 ? 한번 받은 토픽은 영영 받을 수 없는가 ?

    - 발행자는 이 사실을 아는가 ? 알려줄 방법이 있는가 ?

- **sender_overflow.py 코드**
    - Sender라는 이름으로 노드를 생성한다.

    - 발행하는 토픽 이름은 my_topic, type = Int32
    
    - 1초에 1000번 숫자를 1씩 증가시켜 토픽을 발행한다.

- **receiver_overflow.py 코드**
    - Receiver라는 이름으로 노드를 생성한다.

    - 구독자의 콜백 함수 안에 시간이 많이 걸리는 코드를 넣어 토픽 처리에 지연이 생기도록 만든다.

    - Sender로부터 my_topic을 화면에 출력해 토픽의 누락 여부를 확인한다.

    - **1씩 숫자가 증가하지 않으면 문제가 있다는 것을 확인할 수 있다.**


- **현상 확인**
    - **launch 파일 실행 시 받은 토픽이 1씩 증가하지 않는 것을 확인할 수 있다.**

        <img src = 'img/ROS Delay (Problem Situation).png' alt = 'ROS Delay (Problem Situation)' width='500' height='200'>

- **원인 분석**
    - 시간이 많이 걸리는 코드 때문에 **callback할 때 overhead**가 생기고, 제대로 전달 받을 수 없다. 


- **해결책 적용 결과**
    - 받는 쪽 큐 사이즈를 늘린다.
    
    - **버퍼링 덕분에 잃어버리지 않고 콜백을 수행할 수 있게 된다.**

        ```python
        #!/usr/bin/env python

        import rospy
        from std_msgs.msg import Int32

        name = "receiver"
        sub_topic = "my_topic"

        def callback(msg):
            rospy.loginfo("callback is being processed")
            rospy.sleep(5)
            # overhead가 필요하므로 5초동안 일을 하지 않고 쉬도록 만든다. 
            print msg.data

        rospy.init_node(name)
        rospy.Subscriber(sub_topic, Int32, callback, queue_size=10000)
        rospy.spin()
        ```

        <img src = 'img/ROS Delay (Troubleshooting).png' alt = 'ROS Delay (Troubleshooting)' width='500' height='200'>