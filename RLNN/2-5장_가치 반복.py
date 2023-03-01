# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="gEJNiFMuxa6x"
# # 가치 반복

# + id="dL09IfYlxAz3" executionInfo={"status": "ok", "timestamp": 1650110069423, "user_tz": -540, "elapsed": 488, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
import numpy as np
import time
import copy


# + [markdown] id="_QybMndKxW52"
# ## 그림 그리는 함수

# + id="8kriUlj0xWee" executionInfo={"status": "ok", "timestamp": 1650110069925, "user_tz": -540, "elapsed": 8, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
# V table 그리기    
def show_v_table(v_table, env):    
    for i in range(env.reward.shape[0]):        
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                        print("   {0:8.2f}      |".format(v_table[i,j]),end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")

# 정책 policy 화살표로 그리기
def show_policy(policy,env):
    for i in range(env.reward.shape[0]):        
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("                 |",end="")
                if k==1:
                    if policy[i,j] == 0:
                        print("      ↑         |",end="")
                    elif policy[i,j] == 1:
                        print("      →         |",end="")
                    elif policy[i,j] == 2:
                        print("      ↓         |",end="")
                    elif policy[i,j] == 3:
                        print("      ←         |",end="")
                if k==2:
                    print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")


# + [markdown] id="lgsQsky3w9nQ"
# ## Environment 구현

# + id="j-5_IGGDw6dI" executionInfo={"status": "ok", "timestamp": 1650110069925, "user_tz": -540, "elapsed": 7, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
class Environment():
    
    # 1. 미로밖(절벽), 길, 목적지와 보상 설정
    cliff = -3
    road = -1
    goal = 1
    
    # 2. 목적지 좌표 설정
    goal_position = [2,2]
    
    # 3. 보상 리스트 숫자
    reward_list = [[road,road,road],
                   [road,road,road],
                   [road,road,goal]]
    
    # 4. 보상 리스트 문자
    reward_list1 = [["road","road","road"],
                    ["road","road","road"],
                    ["road","road","goal"]]
    
    # 5. 보상 리스트를 array로 설정
    def __init__(self):
        self.reward = np.asarray(self.reward_list)    

    # 6. 선택된 에이전트의 행동 결과 반환 (미로밖일 경우 이전 좌표로 다시 복귀)
    def move(self, agent, action):
        
        done = False
        
        # 6.1 행동에 따른 좌표 구하기
        new_pos = agent.pos + agent.action[action]
        
        # 6.2 현재좌표가 목적지 인지확인
        if self.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        # 6.3 이동 후 좌표가 미로 밖인 확인    
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        # 6.4 이동 후 좌표가 길이라면
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0],observation[1]]
            
        return observation, reward, done


# + [markdown] id="tT7seyk7xIBs"
# ## Agent 구현

# + id="XK7_RCYNxJyr" executionInfo={"status": "ok", "timestamp": 1650110069926, "user_tz": -540, "elapsed": 7, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
class Agent():
    
    # 1. 행동에 따른 에이전트의 좌표 이동(위, 오른쪽, 아래, 왼쪽) 
    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    
    # 2. 각 행동별 선택확률
    select_action_pr = np.array([0.25,0.25,0.25,0.25])
    
    # 3. 에이전트의 초기 위치 저장
    def __init__(self):
        self.pos = (0,0)
    
    # 4. 에이전트의 위치 저장
    def set_pos(self,position):
        self.pos = position
        return self.pos
    
    # 5. 에이전트의 위치 불러오기
    def get_pos(self):
        return self.pos


# + [markdown] id="-RrVAIaox27F"
# ## 가치 반복 함수

# + id="OopZxTb4wSTK" executionInfo={"status": "ok", "timestamp": 1650110069926, "user_tz": -540, "elapsed": 7, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
def finding_optimal_value_function(env, agent, v_table):
    k = 1
    gamma = 0.9
    while(True):
        # Δ←0
        delta=0
        #  v←𝑉(𝑠)
        temp_v = copy.deepcopy(v_table)

        # 모든 𝑠∈𝑆에 대해 :
        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                temp = -1e+10
#                 print("s({0}):".format(i*env.reward.shape[0]+j))
                # 𝑉(𝑠)← max(a)⁡∑𝑃(𝑠'|𝑠,𝑎)[𝑟(𝑠,𝑎,𝑠') +𝛾𝑉(𝑠')]
                # 가능한 행동을 선택
                for action in range(len(agent.action)):
                    agent.set_pos([i,j])
                    observation, reward, done = env.move(agent, action)
#                     print("{0:.2f} = {1:.2f} + {2:.2f} * {3:.2f}" .format(reward + gamma* v_table[observation[0],observation[1]],reward, gamma,v_table[observation[0],observation[1]]))
                    #이동한 상태의 가치가 temp보다 크면
                    if temp < reward + gamma*v_table[observation[0],observation[1]]:
                        # temp 에 새로운 가치를 저장
                        temp = reward + gamma*v_table[observation[0],observation[1]]  
#                 print("V({0}) :max = {1:.2f}".format(i*env.reward.shape[0]+j,temp))
#                 print()
                # 이동 가능한 상태 중 가장 큰 가치를 저장
                v_table[i,j] = temp

        #  ∆←max⁡(∆,|v−𝑉(𝑠)|)
        # 이전 가치와 비교해서 큰 값을 delta에 저장
        # 계산전과 계산후의 가치의 차이 계산
        delta = np.max([delta, np.max(np.abs(temp_v-v_table))])  
        # 7. ∆ <𝜃가 작은 양수 일 때까지 반복
        if delta < 0.0000001:
            break
            
#         if k < 4 or k > 150:
#             print("V{0}(S) : k = {1:3d}    delta = {2:0.6f}".format(k,k, delta))
#             show_v_table(np.round(v_table,2),env)
        print("V{0}(S) : k = {1:3d}    delta = {2:0.6f}".format(k,k, delta))
        show_v_table(np.round(v_table,2),env)
        k +=1
        
    return v_table

def policy_extraction(env, agent, v_table, optimal_policy):

    gamma = 0.9
    
    #정책 𝜋를 다음과 같이 추출
    # 𝜋(𝑠)← argmax(a)⁡∑𝑃(𝑠'|𝑠,𝑎)[𝑟(𝑠,𝑎,𝑠') +𝛾𝑉(𝑠')]
    # 모든 𝑠∈𝑆에 대해 : 
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            temp =  -1e+10
            # 가능한 행동중 가치가 가장높은 값을 policy[i,j]에 저장
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent,action)
                if temp < reward + gamma * v_table[observation[0],observation[1]]:
                    optimal_policy[i,j] = action
                    temp = reward + gamma * v_table[observation[0],observation[1]]
                
    return optimal_policy



# + colab={"base_uri": "https://localhost:8080/"} id="Y90PiuNxwwWm" executionInfo={"status": "ok", "timestamp": 1650110070968, "user_tz": -540, "elapsed": 1048, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}} outputId="789297e4-32e6-4070-aade-8238e4c23036"
# 가치 반복

# 환경, 에이전트를 초기화
np.random.seed(0)
env = Environment()
agent = Agent()

# 초기화
# 모든 𝑠∈𝑆^+에 대해 𝑉(𝑠)∈𝑅을 임의로 설정
v_table =  np.random.rand(env.reward.shape[0], env.reward.shape[1])

print("Initial random V0(S)")
show_v_table(np.round(v_table,2),env)
print()

optimal_policy = np.zeros((env.reward.shape[0], env.reward.shape[1]))

print("start Value iteration")
print()

# 시작 시간 변수에 저장
start_time = time.time()

v_table = finding_optimal_value_function(env, agent, v_table)

optimal_policy = policy_extraction(env, agent, v_table, optimal_policy)

                
print("total_time = {}".format(np.round(time.time()-start_time),2))
print()
print("Optimal policy")
show_policy(optimal_policy, env)
