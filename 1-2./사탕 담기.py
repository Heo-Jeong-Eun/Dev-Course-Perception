'''
💡 approaches
완전 탐색, 모든 경우의 수를 확인하기 위해 combinations를 사용을 생각했다. 

🔑 quiz solution
1. for문을 weights 길이만큼 반복
2. weights에서 추출하는 갯수를 조절하며 모든 경우의 수를 comb 리스트에 저장한다. 
3. comb 리스트에 저장된 요소의 합이 m과 같다면 answer를 ++, answer에 저장된 값을 return 
'''

from itertools import combinations

def solution(m, weights):
    answer = 0

    for i in range(len(weights)):
        comb = list(combinations(weights, i))
        for j in comb:
            if m == sum(j):
                answer += 1

    return answer

m = 3000
weights = [500, 1500, 2500, 1000, 2000]

print(solution(m, weights)) 