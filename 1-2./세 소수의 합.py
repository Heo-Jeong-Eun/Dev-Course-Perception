'''
💡 approaches
에라토스테네스의 체로 소수 판별 

🔑 quiz solution
1. 에라토스테네스의 체를 사용해 소수를 판별한다.
2. combinations 사용해 완전 탐색으로 3개의 수를 sum, n이 되는지 검사하고 n의 count를 반환한다.   

'''
import math
from itertools import combinations

def prime_number(n):
    check = [True] * n # 체로 거르기 전 모든 수 나열
    
    for i in range(2, int(n ** 0.5) + 1): 
    # 최대 배수를 선정하는 기준은 int(n ** 0.5)를 넘기지 않는다. 
    # 구하고자 하는 수까지 반복 낭비를 하지 않기 위해, 나열된 수를 제거하는 최대 배수는 n의 제곱근 이하
        if check[i] == True:
            for j in range(2 * i, n, i): # 배수 자기 자신은 남겨야하므로 2 * i부터 i씩 커지면서 소수가 아닌 수를 제거 
                check[j] = False # 합성수의 자리는 False, 배수의 자리는 True로 남는다. 
    
    return [i for i in range(2, n) if check[i] == True]


def solution(n):
    primes = prime_number(n)
 
    return [sum(i) for i in combinations(primes, 3)].count(n)