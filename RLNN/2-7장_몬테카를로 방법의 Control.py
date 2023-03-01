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
# # 몬테카를로 방법의 Control

# + executionInfo={"elapsed": 359, "status": "ok", "timestamp": 1651296571271, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="dL09IfYlxAz3"
import numpy as np
from tqdm import tqdm


# + [markdown] id="_QybMndKxW52"
# ## 그림 그리는 함수

# + executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1651296571969, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="8kriUlj0xWee"
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

# + executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1651296571970, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="j-5_IGGDw6dI"
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

# + executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1651296571970, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="XK7_RCYNxJyr"
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

# + executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1651296571971, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="OopZxTb4wSTK"
def generate_episode_with_policy(env, agent, first_visit, policy):
    gamma = 0.09
    # 에피소드를 저장할 리스트
    episode = []
    # 이전에 방문여부 체크
    visit = np.zeros((env.reward.shape[0], env.reward.shape[1],len(agent.action)))
    
    # 에이전트는 항상 (0,0)에서 출발
    i = 0
    j = 0
    agent.set_pos([i,j])    
    #에피소드의 수익을 초기화
    G = 0
    #감쇄율의 지수
    step = 0
    max_step = 100
    # 에피소드 생성
    for k in range(max_step):
        pos = agent.get_pos()        
        # 현재 상태의 정책을 이용해 행동을 선택한 후 이동
        action = np.random.choice(range(0,len(agent.action)), p=policy[pos[0],pos[1],:]) 
        observaetion, reward, done = env.move(agent, action)    
        
        if first_visit:
            # 에피소드에 첫 방문한 상태인지 검사 :
            # visit[pos[0],pos[1]] == 0 : 첫 방문
            # visit[pos[0],pos[1]] == 1 : 중복 방문
            if visit[pos[0],pos[1],action] == 0:   
                # 에피소드가 끝날때까지 G를 계산
                G += gamma**(step) * reward        
                # 방문 이력 표시
                visit[pos[0],pos[1],action] = 1
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
# ## ϵ-정책을 이용하는 몬테카를로 control 알고리즘

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3162, "status": "ok", "timestamp": 1651296575122, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}, "user_tz": -540} id="Y90PiuNxwwWm" outputId="ed7d6d9f-4a10-4db4-8d0f-ed4f7dee3f8e"
np.random.seed(0)
# 환경, 에이전트를 초기화
env = Environment()
agent = Agent()

# 모든 𝑠∈𝑆,𝑎∈𝐴(𝑆)에 대해 초기화:
# # 𝑄(𝑠,𝑎)←임의의 값 (행동 개수, 미로 세로, 미로 가로)
Q_table = np.random.rand(env.reward.shape[0], env.reward.shape[1],len(agent.action))
print("Initial Q(s,a)")
show_q_table(Q_table,env)

# 상태를 방문한 횟수를 저장하는 테이블
Q_visit = np.zeros((env.reward.shape[0], env.reward.shape[1],len(agent.action)))

# 미로 모든 상태에서 최적 행동을 저장하는 테이블
# 각 상태에서 Q 값이 가장 큰 행동을 선택 후 optimal_a 에 저장
optimal_a = np.zeros((env.reward.shape[0],env.reward.shape[1]))
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        optimal_a[i,j] = np.argmax(Q_table[i,j,:])
print("initial optimal_a")
show_policy(optimal_a,env)

# π(𝑠,𝑎)←임의의 𝜖−탐욕 정책
# 무작위로 행동을 선택하도록 지정
policy = np.zeros((env.reward.shape[0], env.reward.shape[1],len(agent.action)))

# 한 상태에서 가능한 확률의 합이 1이 되도록 계산
epsilon = 0.8
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        for k in range(len(agent.action)):
            if optimal_a[i,j] == k:
                policy[i,j,k] = 1 - epsilon + epsilon/len(agent.action)
            else:
                policy[i,j,k] = epsilon/len(agent.action)
print("Initial Policy")
show_q_table(policy,env)


# 최대 에피소드 수 길이를 지정
max_episode = 10000

# first visit 를 사용할지 every visit를 사용할 지 결정
# first_visit = True : first visit
# first_visit = False : every visit
first_visit = True
if first_visit:
    print("start first visit MC")
else : 
    print("start every visit MC")
print()

gamma = 0.09
for epi in tqdm(range(max_episode)):
# for epi in range(max_episode):

    # π를 이용해서 에피소드 1개를 생성
    x,y,G,episode = generate_episode_with_policy(env, agent, first_visit, policy)
    
    for step_num in range(len(episode)):
        G = 0
        # episode[step_num][0][0] : step_num번째 방문한 상태의 x 좌표
        # episode[step_num][0][1] : step_num번째 방문한 상태의 y 좌표
        # episode[step_num][1] : step_num번째 상태에서 선택한 행동
        i = episode[step_num][0][0]
        j = episode[step_num][0][1]
        action = episode[step_num][1]
        
        # 에피소드 시작점을 카운트
        Q_visit[i,j,action] += 1

        # 서브 에피소드 (episode[step_num:])의 출발부터 끝까지 수익 G를 계산
        # k[2] : episode[step_num][2] 과 같으며 step_num 번째 받은 보상
        # step : 감쇄율
        for step, k in enumerate(episode[step_num:]):
            G += gamma**(step)*k[2]

        # Incremental mean : 𝑄(𝑠,𝑎)←𝑎𝑣𝑒𝑟𝑎𝑔𝑒(𝑅𝑒𝑡𝑢𝑟𝑛(𝑠,𝑎)) 
        Q_table[i,j,action] += 1 / Q_visit[i,j,action]*(G-Q_table[i,j,action])
    
    # (c) 에피소드 안의 각 s에 대해서 :
    # 미로 모든 상태에서 최적 행동을 저장할 공간 마련
    # 𝑎∗ ←argmax_a 𝑄(𝑠,𝑎)
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            optimal_a[i,j] = np.argmax(Q_table[i,j,:])            
   
    # 모든 𝑎∈𝐴(𝑆) 에 대해서 :
    # 새로 계산된 optimal_a 를 이용해서 행동 선택 확률 policy (π) 갱신
    epsilon = 1 - epi/max_episode

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            for k in range(len(agent.action)):
                if optimal_a[i,j] == k:
                    policy[i,j,k] = 1 - epsilon + epsilon/len(agent.action)
                else:
                    policy[i,j,k] = epsilon/len(agent.action)

print("Final Q(s,a)")
show_q_table(Q_table,env)
print("Final policy")
show_q_table(policy,env)
print("Final optimal_a")
show_policy(optimal_a,env)
