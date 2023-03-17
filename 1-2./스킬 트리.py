'''
💡 approaches
하나의 스킬 트리에서 선행 스킬 순서에 존재하는 것만 s에 추가한다.
만들어진 s의 길이를 기준으로 선행 스킬 순서를 자른다.
선행 스킬 != 스킬 트리, 즉 선행 스킬에 있는 문자가 스킬 트리에 하나도 없다면 답은 1이다. 

🔑 quiz solution
1. 하나의 스킬 트리를 뽑을 때마다 s, index 초기화
2. 스킬 트리에 skill이 있다면 s에 추가한다.
3. 만든 s를 기준으로 skill과 같을 때 cnt += 1
'''

def solution(skill, skill_trees):
    cnt = 0

    for skill_tree in skill_trees:
        s = ''

        for i in skill_tree:
            if i in skill:
                s += i
        
        if skill[:len(s)] == s:
            cnt += 1
    
    return cnt 