# ROS 예제 코드 분석
### student.py

```py
#!/usr/bin/env python
'''
파이썬 코드임을 알려주는 역할을 한다. 실행을 위해서는 파이썬이 필요하다.

#!로 시작되는 라인을 Shebang 라인이라고 한다. 
스크립트 파일의 첫 줄에 사용된다. 
해당 파일의 실행에 어떤 인터프리터를 사용할지 지정한다.
PATH 환경변수에서 가장 우선되는 인터프리터를 찾아 해당 스크립트 파일을 실행한다. 

파이썬의 경우 #으로 시작되는 라인은 주석이므로 소스코드에 미치는 영향은 없다. 
PATH 환경 변수에서 가장 우선시되는 python 바이너리를 찾아 해당 파이썬 파일을 실행한다.
터미널에서 $ python student.py 명령으로 실행시킬 수도 있지만 
Shebang 라인이 들어간 경우에는 $ ./student.py 형식으로 실행할 수 있다.
파이썬 버전을 구분해서 지정할 때에도 사용한다. 
'''

import rospy
'''
rospy라는 라이브러리를 import해서 사용하겠다는 의미이다.
import 키워드는 모듈, 패키지, 파이썬 표준 라이브러리 등을 가져온다. 
모듈 : 특정 기능을 python 파일 단위로 작성한 것
패키지 : 특정 기능과 관련된 여러 모듈을 묶은 것
파이썬 표준 라이브러리 : 파이썬에서 기본적으로 설치된 모듈과 패키지를 묶은 것

rospy는 ROS의 파이썬 클라이언트 라이브러리이다.
rospy를 이용하면 파이썬 프로그래머들이 빠르게 ROS Topic, Services, Parameter의 interface를 사용할 수 있다.
파이썬으로 ROS 프로그래밍을 할 때 필수적인 라이브러리이다.
rospy는 실행속도보다 구현의 편의성을 더 중요하게 생각했기 때문에 빠르게 prototype을 만들 수 있다. 
'''

from std_msgs.msg import String
'''
string msg type을 사용하기 위해 import

from import는 모듈의 일부를 가져올 수 있는 키워드이다. 
from 뒤에 모듈 이름을 지정하고 import 뒤에 가져올 변수, 함수, 클래스를 입력한다.
import 뒤에 여러 개를 넣어도 된다.
import로 모듈을 불러올 때 사용자가 직접 모듈 이름을 설정할 수 있다.
import 뒤에 as를 붙여 이름을 지정하면 된다.
ex) import numpy as np 

위 경우 std_msgs.msg라는 모듈에서 String 관련 부분만 가져와 사용하겠다는 의미이다. 
'''

def callback(msg):
    print msg.data
'''
callback이라는 이름의 함수를 정의한다. 
python에서 함수를 정의하는 방법은 def 키워드를 사용하는 것이다. 
함수를 사용하고자 할 때는 우선 함수를 선언하고 함수를 호출을 원하는 곳에 function_name(parameter) 코드를 작성한다.
함수를 정의할 때는, 함수보다 위에 있어야 한다. 

print msg.data는 msg.data를 화면에 출력하는 코드이다.
rospy는 공식적으로 python2를 사용하고, print문에 괄호를 사용하지 않는다. 
print 작업 시 object들은 string을 내부적으로 바꿔주므로 변수를 넣어도 string으로 변환되어 출력이 가능하다.  
'''
    
rospy.init_node('student')
'''
import 했던 rospy에 있는 init_node를 사용, 노드를 새로 만드는데 이름은 student로 지정한다.

해당 노드를 초기화하고 노드의 이름을 student로 한다는 의미이다.
많은 노드를 관리하고 통합하는 것이 ROS 프레임워크가 하는 일이고, 그것을 python으로 만드는 것이 rospy 라이브러리이다.
init_node는 rospy에서도 기본적인 함수이며 이 함수를 사용해 생성된 노드는 다른 노드와 통신하며 토픽을 주고 받는다.
ROS 시스템 상에서 노드들이 토픽을 주고 받기 위해서는 노드에 고유의 이름을 할당해야 한다. 

init_node() 함수를 자세히 살펴보면
def init_node(name, argv = None, anonymous = False, log_level = None, ...)
첫 번째 인자는 Node의 이름이고 타입은 string이다.
두 번째 인자는 argv인데 사용자가 지정한 argument를 넘겨받을 때 사용하고 타입은 string의 list이다.
세 번째 인자는 anonymous인데 default가 False로 되어 있다. True인 경우 노드의 이름이 자동으로 생성된다.
네 번째 인자는 log_level로 타입은 int이며 default는 INFO level이다. 
다섯 번째와 여섯 번째 인자는 deisable_rostime, disable_rosout으로 내부적인 테스트에서만 사용된다. 
일곱 번째 인자는 disable_signals이다. True의 경우 rospy는 사용자의 signal handler를 등록하지 않는다.
사용자가 main thread로부터 init_node를 call하지 않을 때나, 
혹은 사용자가 자신만이 signal handling을 설정해야하는 환경에서 rospy를 사용할 때 이 flag를 setting 해야 한다. 
여덟 번째 인자는 xmlrpc_port이다. client XMLRPC node에 관한 포트 번호이다.
아홉 번째 인자는 tcpros_port이다. TCPROS server는 이 포트를 통해 통신한다. 
'''

sub = rospy.Subscriber('my_topic', String, callback)
'''
Subscriber임을 선언, 받고자하는 토픽의 이름은 my_topic이고, string type이여야 하는 것을 의미한다. 
토픽이 도착할 때마다 callback 함수를 실행시킬 것을 ROS 시스템에 요청한다. 
결과적으로 토픽을 받을 때마다 callback 함수가 한번씩 호출된다. 
'''

rospy.spin()
'''
무한 루프, 프로그램이 종료되지 않도록 한다.  

ROS 노드가 shutdown 될 때까지 block하는 함수로 shutdown signal을 받을 때까지 무한 루프를 돌린다. 
이 무한 루프에서 토픽을 받거나 time triggering 같은 이벤트가 발생하면 
callback 함수를 호출하고 아닌 경우에는 sleep 상태를 유지한다. 
사용자의 노드가 callback 이외에 어떤 일도 하지 않는다면 spin() 함수를 사용해야 한다.

rospy.spin() ~= rospy.sleep()
rospy.sleep()은 특정 시간이 주어지고 그 시간동안만 sleep 상태를 유지한다. 
'''
```