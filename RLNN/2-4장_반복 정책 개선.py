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
# # 반복 정책 개선

# + executionInfo={"elapsed": 45, "status": "ok", "timestamp": 1650091421764, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="dL09IfYlxAz3"
import numpy as np
import time
import copy


# + [markdown] id="_QybMndKxW52"
# ## 그림 그리는 함수

# + executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1650091449730, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="8kriUlj0xWee"
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

# + executionInfo={"elapsed": 359, "status": "ok", "timestamp": 1650091455046, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="j-5_IGGDw6dI"
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

# + executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1650091457656, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="XK7_RCYNxJyr"
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
# ## 정책 평가 및 개선 함수

# + executionInfo={"elapsed": 354, "status": "ok", "timestamp": 1650091462508, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="OopZxTb4wSTK"
def policy_evalution(env, agent, v_table, policy):
    gamma = 0.9
    while(True):
        # Δ←0
        delta = 0
        #  v←𝑉(𝑠)
        temp_v = copy.deepcopy(v_table)
        # 모든 𝑠∈𝑆에 대해 :
        for i in range(env.reward.shape[0]):
            for j in range(env.reward.shape[1]):
                # 에이전트를 지정된 좌표에 위치시킨후 가치함수를 계산
                agent.set_pos([i,j])
                # 현재 정책의 행동을 선택
                action = policy[i,j]
                observation, reward, done = env.move(agent, action)
                v_table[i,j] = reward + gamma * v_table[observation[0],observation[1]]
        # ∆←max⁡(∆,|v−𝑉(𝑠)|)
        # 계산전과 계산후의 가치의 차이를 계산
        delta = np.max([delta, np.max(np.abs(temp_v-v_table))])  
                
        # 7. ∆ <𝜃가 작은 양수 일 때까지 반복
        if delta < 0.000001:
            break
    return v_table, delta


def policy_improvement(env, agent, v_table, policy):
    
    # 67페이지 아래 누락 되어있습니다
    gamma = 0.9  
    
    # policyStable ← true 
    policyStable = True

    # 모든 s∈S에 대해：
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):            
            # 𝑜𝑙𝑑−𝑎𝑐𝑡𝑖𝑜𝑛←π(s) 
            old_action = policy[i,j]            
            # 가능한 행동중 최댓값을 가지는 행동을 선택
            temp_action = 0
            temp_value =  -1e+10           
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent,action)
                if temp_value < reward + gamma * v_table[observation[0],observation[1]]:
                    temp_action = action
                    temp_value = reward + gamma * v_table[observation[0],observation[1]]
            # 만약 𝑜𝑙𝑑−𝑎𝑐𝑡𝑖𝑜𝑛"≠π(s)"라면， "policyStable ← False" 
            # old-action과 새로운 action이 다른지 체크
            if old_action != temp_action :
                policyStable = False
            policy[i,j] = temp_action
    return policy, policyStable



# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 331, "status": "ok", "timestamp": 1650091466516, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="Y90PiuNxwwWm" outputId="f43dec87-6813-4b09-f471-99b27723d479"
# 정책 반복
# 환경과 에이전트에 대한 초기 설정
np.random.seed(0)
env = Environment()
agent = Agent()

# 1. 초기화
# 모든 𝑠∈𝑆에 대해 𝑉(𝑠)∈𝑅과 π(𝑠)∈𝐴(𝑠)를 임의로 설정
v_table =  np.random.rand(env.reward.shape[0], env.reward.shape[1])
policy = np.random.randint(0, 4,(env.reward.shape[0], env.reward.shape[1]))

print("Initial random V(S)")
show_v_table(np.round(v_table,2),env)
print()
print("Initial random Policy π0(S)")
show_policy(policy,env)
print("start policy iteration")

# 시작 시간을 변수에 저장
start_time = time.time()

max_iter_number = 20000
for iter_number in range(max_iter_number):
    
    # 2.정책평가
    v_table, delta = policy_evalution(env, agent, v_table, policy)

    # 정책 평가 후 결과 표시                                            
    print("")
    print("Vπ{0:}(S) delta = {1:.10f}".format(iter_number,delta))
    show_v_table(np.round(v_table,2),env)
    print()    
    
    
    # 3.정책개선
    policy, policyStable = policy_improvement(env, agent, v_table, policy)

    # policy 변화 저장
    print("policy π{}(S)".format(iter_number+1))
    show_policy(policy,env)
    # 하나라도 old-action과 새로운 action이 다르다면 '2. 정책평가'를 반복
    if(policyStable == True):
        break

        
print("total_time = {}".format(time.time()-start_time))
