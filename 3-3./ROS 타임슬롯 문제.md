- ## **주기적 발송에서 타임 슬롯을 오버하면 어떻게 되는가 ?**

- **teacher_int32_job.py**
    
    ```python
    #!/usr/bin/env python

    import rospy
    import time
    from std_msgs.msg import Int32

    def do_job(count):
        for i in range(0, count):
            i += 1
            # 단순하게 정수값을 증가시켜 publish
            pub.publish(i)

    def list_append_time():
        start.append(start_time)
        end.append(end_time)
        sleep.append(sleep_time)
    # job의 시작, 끝, 쉬는 시간을 list에 추가

    rospy.init_node('teacher')
    pub = rospy.Publisher('msg_to_students', Int32, queue_size=0)
    rate = rospy.Rate(5)
    # 초당 5번, 즉 0.2초

    while not rospy.is_shutdown():
        start = []
        end = []
        sleep = []

        num = input("input count number > ")
        
        rate.sleep()

        total_start = time.time()

        for j in range(0, 5):
            start_time = time.time()
            do_job(num)
            end_time = time.time()

            rate.sleep()
            sleep_time = time.time()
            list_append_time()

    total_end = time.time()

    for t in range(0, 5):
        sleep[t] -= end[t]
        end[t] -= start[t]

    for result in range(0, 5):
        print "spend time > ", round(end[result], 4), 's'
        print "sleep time > ", round(sleep[result], 4), 's'

    print "------------------------------"
    print "total time > ", round((total_end - total_start), 4), 's'
    print "------------------------------\n\n"
    ```


- 1초에 5번 반복하게 하고 작업시간이 0.2초가 넘어가게 만든다.
    - Rate(5)로 설정하고 sleep() 앞에 들어간 작업 코드에 대해 수행 시간을 늘린다.

    - 늘렸다가 줄였다가 변동성 있게 한다. 입력 값을 받아서 조정할 수 있게 만들어본다.

- 1초에 5번 규칙을 지킬 수 없으면 어떻게 할까 ?
    - 앞에서부터 쭉 밀리는 식으로 일을 할까 ?

    - 쉬는 시간을 조정할까 ?

    - 이번에는 3번만 하고 다음번을 기약할까 ?

- 토픽 이름은 msg_to_student, type = Int32

- 0.2초에 다섯 차례 토픽을 보낸다.
    - rate = rospy.Rate(5), rate.sleep()

- 키보드 입력을 받던지, launch parameter를 이용해 한번 보낼 양을 설정한다.
    - 100이면 0.2초에 타임 슬롯에 1, 2, 3, …100을 보낸다.

- 시간 함수 time.time()을 이용해 각 슬롯 간 소요 시간과 5개의 전체 슬롯의 소요 시간을 계산해 출력한다.


- **현상 확인**
    - 1초에 5번 작업, count 값이 1000인 경우 

        <img src = 'img/ROS TimeSlot 1000.png' alt = 'ROS TimeSlot 1000' width='500' height='200'>

    - 1초에 5번 작업, count 값이 4000인 경우 

        <img src = 'img/ROS TimeSlot 4000.png' alt = 'ROS TimeSlot 4000' width='500' height='200'>

    - 1초에 5번 작업, count 값이 1500인 경우 

        <img src = 'img/ROS TimeSlot 1500.png' alt = 'ROS TimeSlot 1500' width='500' height='200'>

    - 1초에 5번 작업, count 값이 1800인 경우 

        <img src = 'img/ROS TimeSlot 1800.png' alt = 'ROS TimeSlot 1800' width='500' height='200'>

    - 1초에 5번 작업, count 값이 50000인 경우 

        <img src = 'img/ROS TimeSlot 50000.png' alt = 'ROS TimeSlot 50000' width='500' height='200'>


- **원인 분석**
    - **타임 슬롯 안에 할 일을 마치지 못하는 경우는 하는 일의 양이 아주 적거나, 많을 때 발생한다.**

    - **한 타임 슬롯에서 한번의 job과 Rate / sleep()이 발생하는 것이 기본이다.** 

        - 주어진 타임 슬롯에서 자신이 할 일을 마치고, 시간이 남으면 잠시 휴식 시간을 갖는다. 

    - 일의 양을 키우는 방법으로 타임 슬롯보다 job이 커지도록 유도 → 타임 슬롯이 오버하는 상황을 만든다. 