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

# + [markdown] id="5agojReWIYlV"
# # 시간차 학습의 Prediction

# + executionInfo={"elapsed": 337, "status": "ok", "timestamp": 1651652285485, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="CwtfaBKOhqaB"
import numpy as np
from tqdm import tqdm


# + [markdown] id="JQkVjrbvI2_s"
# ## 그림그리는 함수

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1651652285825, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="-maf_uhdI2k9"
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


# + [markdown] id="8EIdUJwmIgFS"
# ## Environment 구현

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1651652285825, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="HytMoTUaeV4-"
class Environment():
    
    # 보상 설정
    cliff = -3
    road = -1
    goal = 1

    reward_list = [[road,road,road],
                   [road,road,road],
                   [road,road,goal]]

    reward_list1 = [["road","road","road"],
                    ["road","road","road"],
                    ["road","road","goal"]]

    def __init__(self):
        self.reward = np.array(self.reward_list)
     
    def move(self, agent, action):
        
        done = False

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
      


# + [markdown] id="eNm5b3QNIhjx"
# ## Agent 구현

# + executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1651652286252, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="KApr6k3LiJmF"
class Agent():

    action = np.array([[-1,0],[0,1],[1,0],[0,-1]])
    
    select_action_pr = np.array([0.25,0.25,0.25,0.25])

    def __init__(self):
        self.pos = (0,0)
    
    def set_pos(self,position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos


# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 366, "status": "ok", "timestamp": 1651652286610, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="tBxFRf1JIUHU" outputId="90b10be8-a688-45a3-b203-febac49c0680"
# TD(0) prediction
np.random.seed(0)
# 환경, 에이전트를 초기화
env = Environment()
agent = Agent()
gamma = 0.9

#초기화 : 
#π← 평가할 정책
# 가능한 모든 행동이 무작위로 선택되도록 지정
#𝑉← 임의의 상태가치 함수
V = np.zeros((env.reward.shape[0], env.reward.shape[1]))

# 최대 에피소드, 에피소드의 최대 길이를 지정
max_episode = 10000
max_step = 100

alpha = 0.01

print("start TD(0) prediction")

# 각 에피소드에 대해 반복 :
for epi in tqdm(range(max_episode)):
    delta =0
    # s 를 초기화
    i = 0
    j = 0
    agent.set_pos([i,j])

    #  에피소드의 각 스텝에 대해 반복 :
    for k in range(max_step):
        pos = agent.get_pos()
        # a←상태 𝑠 에서 정책 π에 의해 결정된 행동 
        # 가능한 모든 행동이 무작위로 선택되게 함
        action = np.random.randint(0,4)
        # 행동 a 를 취한 후 보수 r과 다음 상태 s’를 관측
        # s←𝑠'
        observation, reward, done = env.move(agent,action)
        # V(𝑠)←V(𝑠)+ α[𝑟+𝛾𝑉(𝑠^)−𝑉(𝑠)]
        V[pos[0],pos[1]] += alpha * (reward + gamma * V[observation[0],observation[1]] - V[pos[0],pos[1]])
        # s가 마지막 상태라면 종료
        if done == True:
            break
            
print("V(s)")
show_v_table(np.round(V,2),env)
