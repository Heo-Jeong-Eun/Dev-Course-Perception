'''
💡 approaches
queue를 활용한 풀이

🔑 quiz solution
1. cnt, t(time) 변수 선언
2. index 0번 요소가 100이 될 때까지 반복하며 t를 증가시킨다.
3. 100을 만족하면 요소를 pop, cnt++
4. 100이 되지 않는 경우 이전에 완료된 cnt는 answer에 append하고 cnt를 초기화 시킨다. 
5. t가 증가해 100이 넘으면 cnt++, answer에 append한다. 
'''

def solution(progresses, speeds):
    answer = []
    t = 0
    cnt = 0

    while len(progresses) > 0:
        if (progresses[0] + speeds[0] * t) >= 100:
            progresses.pop(0)
            speeds.pop(0)
            cnt += 1
        else:
            if cnt > 0:
                answer.append(cnt)
                cnt = 0
            else:
                t += 1
    answer.append(cnt)

    return answer