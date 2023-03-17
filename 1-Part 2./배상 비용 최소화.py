'''
💡 approaches
Heap 자료구조 활용, 우선순위 큐 
파이썬의 Heapq 모듈은 MinHeap을 기반으로 구현되었기 때문에 이를 응용해 MaxHeap을 사용한다.

🔑 quiz solution
1. (-value, value)로 원소를 지정, (우선순위, 실제값)으로 사용한다.
2. works 배열을 MaxHeap으로 변환한 뒤, no만큼 순회하며 최대값을 추출, -1 연산을 진행
3. 이후 MaxHeap의 모든 원소의 첫번째 원소를 뽑아 해당 값의 제곱을 모두 더해 반환한다. 
'''

import heapq

def solution(no, works):
    MaxHeap = []

    if no >= sum(works):
        return 0
    
    for work in works:
        heapq.heappush(MaxHeap, (-work, work))

    for _ in range(no):
        work = heapq.heappop(MaxHeap)[1] - 1
        heapq.heappush(MaxHeap, (-work, work))

    return sum([i[1] ** 2 for i in MaxHeap])

N = 4
works = [4, 3, 3]

print(solution(N, works))