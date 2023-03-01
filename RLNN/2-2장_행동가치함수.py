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

# + [markdown] id="h794qK3Oic0h"
# # 행동 가치 함수

# + id="cIrr-8sBi8sj" executionInfo={"status": "ok", "timestamp": 1649310192600, "user_tz": -540, "elapsed": 517, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
import numpy as np


# + [markdown] id="wLW0z-vOiiBh"
# ## 그림그리는 함수

# + id="K0mDS_iAjP7D" executionInfo={"status": "ok", "timestamp": 1649310289944, "user_tz": -540, "elapsed": 371, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
# Q table 그리기
def show_q_table(q_table,env):
    for i in range(env.reward.shape[0]):
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    print("{0:10.2f}       |".format(q_table[i,j,0]),end="")
                if k==1:
                    print("{0:6.2f}    {1:6.2f} |".format(q_table[i,j,3],q_table[i,j,1]),end="")
                if k==2:
                    print("{0:10.2f}       |".format(q_table[i,j,2]),end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")

# 정책 policy 화살표로 그리기
def show_q_table_arrow(q_table,env):
    for i in range(env.reward.shape[0]):        
        print("+-----------------"*env.reward.shape[1],end="")
        print("+")
        for k in range(3):
            print("|",end="")
            for j in range(env.reward.shape[1]):
                if k==0:
                    if np.max(q[i,j,:]) == q[i,j,0]:
                        print("        ↑       |",end="")
                    else:
                        print("                 |",end="")
                if k==1:                    
                    if np.max(q[i,j,:]) == q[i,j,1] and np.max(q[i,j,:]) == q[i,j,3]:
                        print("      ←  →     |",end="")
                    elif np.max(q[i,j,:]) == q[i,j,1]:
                        print("          →     |",end="")
                    elif np.max(q[i,j,:]) == q[i,j,3]:
                        print("      ←         |",end="")
                    else:
                        print("                 |",end="")
                if k==2:
                    if np.max(q[i,j,:]) == q[i,j,2]:
                        print("        ↓       |",end="")
                    else:
                        print("                 |",end="")
            print()
    print("+-----------------"*env.reward.shape[1],end="")
    print("+")    


# + [markdown] id="hVxMbO0_ip7c"
# ## Environment 구현

# + id="hNHgmsNjiM1T" executionInfo={"status": "ok", "timestamp": 1649310198114, "user_tz": -540, "elapsed": 366, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
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


# + [markdown] id="odBKf64tikWQ"
# ## Agent 구현

# + id="WOQE9pHPiWpR" executionInfo={"status": "ok", "timestamp": 1649310201579, "user_tz": -540, "elapsed": 358, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
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


# + [markdown] id="pGYH8lvAh0jf"
# ## 행동 가치 함수

# + id="LnidkLh8hSrr" executionInfo={"status": "ok", "timestamp": 1649310204526, "user_tz": -540, "elapsed": 521, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
# 재귀적으로 행동가치함수를 계산하는 함수

# 행동 가치 함수
def action_value_function(env, agent, act, G, max_step, now_step):   
    
    # 1. 감가율 설정
    gamma = 0.9
    
    # 2. 현재 위치가 목적지인지 확인
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
        return env.goal

    # 3. 마지막 상태는 보상만 계산
    if (max_step == now_step):
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act]*reward
        return G
    
    # 4. 현재 상태의 보상을 계산한 후 다음 행동과 함께 다음 step으로 이동
    else:
        # 4.1현재 위치 저장
        pos1 = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act] * reward
        
        # 4.2 이동 후 위치 확인 : 미로밖, 벽, 구멍인 경우 이동전 좌표로 다시 이동
        if done == True:            
            if observation[0] < 0 or observation[0] >= env.reward.shape[0] or observation[1] < 0 or observation[1] >= env.reward.shape[1]:
                agent.set_pos(pos1)
            
        # 4.3 현재 위치를 다시 저장
        pos1 = agent.get_pos()
        
        # 4.4 현재 위치에서 가능한 모든 행동을 선택한 후 이동
        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            next_v = action_value_function(env, agent, i, 0, max_step, now_step+1)
            G += agent.select_action_pr[i] * gamma * next_v
        return G


# + [markdown] id="4Oz-zwfZh72X"
# ## 미로의 각 상태의 행동가치함수를 구하는 함수

# + colab={"base_uri": "https://localhost:8080/"} id="oTqhVC3ih5HW" executionInfo={"status": "ok", "timestamp": 1649310298786, "user_tz": -540, "elapsed": 4262, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}} outputId="89e9f5e4-22fa-4e2b-d926-252339c93d12"
# 재귀적으로 행동의 가치를 계산

# 1. 환경 초기화
env = Environment()

# 2. 에이전트 초기화
agent = Agent()
np.random.seed(0)

# 3. 현재부터 max_step 까지 계산
max_step_number = 8

# 4. 모든 상태에 대해
for max_step in range(max_step_number):
    # 4.1 미로 상의 모든 상태에서 가능한 행동의 가치를 저장할 테이블을 정의
    print("max_step = {}".format(max_step))
    q_table = np.zeros((env.reward.shape[0], env.reward.shape[1],len(agent.action)))
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            # 4.2 모든 행동에 대해
            for action in range(len(agent.action)):
                # 4.2.1 에이전트의 위치를 초기화
                agent.set_pos([i,j])
                # 4.2.2 현재 위치에서 행동 가치를 계산
                q_table[i ,j,action] = action_value_function(env, agent, action, 0, max_step, 0)

    q = np.round(q_table,2)
    print("Q - table")
    show_q_table(q, env)
    print("High actions Arrow")
    show_q_table_arrow(q,env)
    print()
