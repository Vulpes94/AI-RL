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
# # 몬테카를로 방법의 Prediction

# + executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1651652211296, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="dL09IfYlxAz3"
import numpy as np
from tqdm import tqdm


# + [markdown] id="_QybMndKxW52"
# ## 그림 그리는 함수

# + executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1651652211296, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="8kriUlj0xWee"
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


# + [markdown] id="lgsQsky3w9nQ"
# ## Environment 구현

# + executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1651652211879, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="j-5_IGGDw6dI"
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

# + executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1651652211880, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="XK7_RCYNxJyr"
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
# ## 에피소드 생성 함수

# + executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1651652211881, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="OopZxTb4wSTK"
def generate_episode(env, agent, first_visit):
    gamma = 0.09
    # 에피소드를 저장할 리스트
    episode = []
    # 이전에 방문여부 체크
    visit = np.zeros((env.reward.shape[0], env.reward.shape[1]))
    
    # 에이전트가 모든 상태에서 출발할 수 있게 출발지점을 무작위로 설정
    i = np.random.randint(0,env.reward.shape[0])
    j = np.random.randint(0,env.reward.shape[1])
    agent.set_pos([i,j])    
    #에피소드의 수익을 초기화
    G = 0
    #감쇄율의 지수
    step = 0
    max_step = 100
    # 에피소드 생성
    for k in range(max_step):
        pos = agent.get_pos()            
        action = np.random.randint(0,len(agent.action))            
        observaetion, reward, done = env.move(agent, action)    
        
        if first_visit:
            # 에피소드에 첫 방문한 상태인지 검사 :
            # visit[pos[0],pos[1]] == 0 : 첫 방문
            # visit[pos[0],pos[1]] == 1 : 중복 방문
            if visit[pos[0],pos[1]] == 0:   
                # 에피소드가 끝날때까지 G를 계산
                G += gamma**(step) * reward        
                # 방문 이력 표시
                visit[pos[0],pos[1]] = 1
                step += 1               
                # 방문 이력 저장(상태, 행동, 보상)
                episode.append((pos,action, reward))
        else:
            G += gamma**(step) * reward
            step += 1                   
            episode.append((pos,action,reward))            

        # 에피소드가 종료했다면 루프에서 탈출
        if done == True:                
            break        
            
    return i, j, G, episode


# + [markdown] id="gQtwLOM2qpLZ"
# ## First-visit and Every-Visit MC Prediction

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 59318, "status": "ok", "timestamp": 1651652271187, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="Y90PiuNxwwWm" outputId="5897567a-5771-407c-883b-0912725a13ae"
# first-visit MC and every-visit MC prediction
np.random.seed(0)
# 환경, 에이전트를 초기화
env = Environment()
agent = Agent()

# 임의의 상태 가치 함수𝑉
v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))

# 상태별로 에피소드 출발횟수를 저장하는 테이블
v_start = np.zeros((env.reward.shape[0], env.reward.shape[1]))

# 상태별로 도착지점 도착횟수를 저장하는 테이블
v_success = np.zeros((env.reward.shape[0], env.reward.shape[1]))

# 𝑅𝑒𝑡𝑢𝑟𝑛(𝑠)←빈 리스트 (모든 s∈𝑆에 대해)
Return_s = [[[] for j in range(env.reward.shape[1])] for i in range(env.reward.shape[0])]

# 최대 에피소드 수를 지정
max_episode = 100000

# first visit 를 사용할지 every visit를 사용할 지 결정
# first_visit = True : first visit
# first_visit = False : every visit
first_visit = False
if first_visit:
    print("start first visit MC")
else : 
    print("start every visit MC")
print()

for epi in tqdm(range(max_episode)):
    
    i,j,G,episode = generate_episode(env, agent, first_visit)
    
    # 수익 𝐺를 𝑅𝑒𝑡𝑢𝑟𝑛(𝑠)에 추가(append)
    Return_s[i][j].append(G)
    
    # 에피소드 발생 횟수 계산
    episode_count = len(Return_s[i][j])
    # 상태별 발생한 수익의 총합 계산
    total_G = np.sum(Return_s[i][j])
    # 상태별 발생한 수익의 평균 계산
    v_table[i,j] = total_G / episode_count
    
  # 도착지점에 도착(reward = 1)했는지 체크    
    # episode[-1][2] : 에피소드 마지막 상태의 보상
    if episode[-1][2] == 1:
        v_success[i,j] += 1

# 에피소드 출발 횟수 저장 
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        v_start[i,j] = len(Return_s[i][j])
        
print("V(s)")
show_v_table(np.round(v_table,2),env)
print("V_start_count(s)")
show_v_table(np.round(v_start,2),env)
print("V_success_pr(s)")
show_v_table(np.round(v_success/v_start,2),env)
