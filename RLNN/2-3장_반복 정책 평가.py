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

# + [markdown] id="i0tVKAcrcqRh"
# # 반복 정책 평가

# + id="--QFD6ascmgH" executionInfo={"status": "ok", "timestamp": 1649862301095, "user_tz": -540, "elapsed": 470, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
import numpy as np
import time
import copy


# + [markdown] id="Rposyny5c-0s"
# ## 그림 그리는 함수

# + id="2rtdYis1c9KB" executionInfo={"status": "ok", "timestamp": 1649862263377, "user_tz": -540, "elapsed": 316, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
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


# + [markdown] id="Tpb0_es0cwg5"
# ## Environment 구현

# + id="wPtI__gUcbNg" executionInfo={"status": "ok", "timestamp": 1649862156068, "user_tz": -540, "elapsed": 4, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
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


# + [markdown] id="r18dfVIxc4IB"
# ## Agent 구현

# + id="jc2kYfpzce_l" executionInfo={"status": "ok", "timestamp": 1649862205270, "user_tz": -540, "elapsed": 320, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
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


# + [markdown] id="GcEnEcXybtWb"
# 반복 정책 평가

# + id="BPzBnyNSbf7K" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1649862307262, "user_tz": -540, "elapsed": 864, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}} outputId="022b2ee2-89b1-45a1-d1bf-5c0b90e07e01"
# 반복 정책 평가
np.random.seed(0)
env = Environment()
agent = Agent()
gamma = 0.9

# 1. 모든 𝑠∈𝑆^에 대해서 배열 𝑉(𝑠)=0으로 초기화
v_table = np.zeros((env.reward.shape[0],env.reward.shape[1]))

print("start Iterative Policy Evaluation")

k = 1
print()
print("V0(S)   k = 0")

# 초기화된 V 테이블 출력
show_v_table(np.round(v_table,2),env)

# 시작 시간 변수에 저장
start_time = time.time()

# 반복
while(True):    
    # 2. Δ←0
    delta = 0
    # 3. v←(𝑠)
    # 계산전 가치를 저장
    temp_v = copy.deepcopy(v_table)
    # 4. 모든 𝑠∈𝑆에 대해 : 
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            G = 0
            # 5. 가능한 모든 행동으로 다음상태만 이용해 𝑉(𝑠) 계산
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent, action)
                
#                 print("s({0}): {1:5s} : {2:0.2f} = {3:0.2f} *({4:0.2f} +  {5:0.2f} *  {6:0.2f})".format(i*env.reward.shape[0]+j,dic[action],agent.select_action_pr[action] * (reward + gamma*V[observation[0],observation[1]]), agent.select_action_pr[action],reward,gamma,V[observation[0],observation[1]]))

                G += agent.select_action_pr[action] * (reward + gamma*v_table[observation[0],observation[1]])                    

#             print("V{2}({0}) :sum = {1:.2f}".format(i*env.reward.shape[0]+j,total,k))
#             print()
            v_table[i,j] = G
    # 6. ∆←max⁡(∆,|v−𝑉(𝑠)|)
    # 계산전과 계산후의 가치 차이 계산
    delta = np.max([delta, np.max(np.abs(temp_v-v_table))])
    
    end_time = time.time()        
    print("V{0}(S) : k = {1:3d}    delta = {2:0.6f} total_time = {3}".format(k,k, delta,np.round(end_time-start_time),2))
    show_v_table(np.round(v_table,2),env)                
    k +=1

    # 7. ∆ <𝜃가 작은 양수 일 때까지 반복

    if delta < 0.000001:
        break
        
end_time = time.time()        
print("total_time = {}".format(np.round(end_time-start_time),2))
