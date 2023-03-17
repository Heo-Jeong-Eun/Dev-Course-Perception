'''
💡 approaches
product 활용, 두 개 이상의 리스트이 모든 조합을 구하기 위해 사용한다. 

🔑 quiz solution
1. 각각의 주사위마다 나올 수 있는 값을 리스트로 만든다.
2. 리스트를 product한 원소의 합을 구한다. 
3. product를 탐색하며 monster가 있는 칸에 도달하면 cnt++
4. 1 - cnt 확률을 반환한다. 
'''

from itertools import product

def solution(monster, S1, S2, S3):
    l1 = list(range(1, S1 + 1))
    l2 = list(range(1, S2 + 1))
    l3 = list(range(1, S3 + 1))
    
    prod = list(map(sum, product(l1, l2, l3)))
    cnt = 0

    for i in prod:
        if i + 1 in monster:
            cnt += 1

    return int((len(prod) - cnt) / len(prod) * 1000)