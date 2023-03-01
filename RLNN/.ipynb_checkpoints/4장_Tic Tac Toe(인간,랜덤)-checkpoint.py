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

# + [markdown] id="RA4gj7vQkjJR"
# # Tic Tac Toe

# + id="q4jheP3BjwS3" executionInfo={"status": "ok", "timestamp": 1649746618132, "user_tz": -540, "elapsed": 353, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
import numpy as np
from tqdm import tqdm


# + [markdown] id="heQAOhM1knUp"
# ## Environment

# + id="jWSQtiKYgy77" executionInfo={"status": "ok", "timestamp": 1649746179970, "user_tz": -540, "elapsed": 4, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
class Environment():
    
    def __init__(self):
    # 보드는 0으로 초기화된 9개의 배열로 준비
    # 게임종료 : done = True
        self.board_a = np.zeros(9)
        self.done = False
        self.reward = 0
        self.winner = 0
        self.print = False

    def move(self, p1, p2, player):
    # 각 플레이어가 선택한 행동을 표시 하고 게임 상태(진행 또는 종료)를 판단
    # p1 = 1, p2 = -1로 정의
    # 각 플레이어는 행동을 선택하는 select_action 메서드를 가짐
        if player == 1:
            pos = p1.select_action(env, player)
        else:
            pos = p2.select_action(env, player)
        
        # 보드에 플레이어의 선택을 표시
        self.board_a[pos] = player
        if self.print:
            print(player)
            self.print_board()
        # 게임이 종료상태인지 아닌지를 판단
        self.end_check(player)
        
        return  self.reward, self.done
 
    # 현재 보드 상태에서 가능한 행동(둘 수 있는 장소)을 탐색하고 리스트로 반환
    def get_action(self):
        observation = []
        for i in range(9):
            if self.board_a[i] == 0:
                observation.append(i)
        return observation
    
    # 게임이 종료(승패 또는 비김)됐는지 판단
    def end_check(self,player):
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # 승패 조건은 가로, 세로, 대각선 이 -1 이나 1 로 동일할 때 
        end_condition = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for line in end_condition:
            if self.board_a[line[0]] == self.board_a[line[1]] \
                and self.board_a[line[1]] == self.board_a[line[2]] \
                and self.board_a[line[0]] != 0:
                # 종료됐다면 누가 이겼는지 표시
                self.done = True
                self.reward = player
                return
        # 비긴 상태는 더는 보드에 빈 공간이 없을때
        observation = self.get_action()
        if (len(observation)) == 0:
            self.done = True
            self.reward = 0            
        return
        
    # 현재 보드의 상태를 표시 p1 = O, p2 = X    
    def print_board(self):
        print("+----+----+----+")
        for i in range(3):
            for j in range(3):
                if self.board_a[3*i+j] == 1:
                    print("|  O",end=" ")
                elif self.board_a[3*i+j] == -1:
                    print("|  X",end=" ")
                else:
                    print("|   ",end=" ")
            print("|")
            print("+----+----+----+")


# + [markdown] id="2Ky3zgEIkrtZ"
# ## Human_player

# + id="uHf4MZERiUKP" executionInfo={"status": "ok", "timestamp": 1649746245363, "user_tz": -540, "elapsed": 379, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
class Human_player():
    
    def __init__(self):
        self.name = "Human player"
        
    def select_action(self, env, player):
        while True:
            # 가능한 행동을 조사한 후 표시
            available_action = env.get_action()
            print("possible actions = {}".format(available_action))

            # 상태 번호 표시
            print("+----+----+----+")
            print("+  0 +  1 +  2 +")
            print("+----+----+----+")
            print("+  3 +  4 +  5 +")
            print("+----+----+----+")
            print("+  6 +  7 +  8 +")
            print("+----+----+----+")
                        
            # 키보드로 가능한 행동을 입력 받음
            action = input("Select action(human) : ")
            action = int(action)
            
            # 입력받은 행동이 가능한 행동이면 반복문을 탈출
            if action in available_action:
                return action
            # 아니면 행동 입력을 반복
            else:
                print("You selected wrong action")
        return


# + [markdown] id="cUMVlkQPkvxb"
# ## Random_player

# + id="CygbohgHixN2" executionInfo={"status": "ok", "timestamp": 1649746561811, "user_tz": -540, "elapsed": 347, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}}
class Random_player():
    
    def __init__(self):
        self.name = "Random player"
        
    def select_action(self, env, player):
        # 가능한 행동 조사
        available_action = env.get_action()
        # 가능한 행동 중 하나를 무작위로 선택
        action = np.random.randint(len(available_action))

        return available_action[action]


# + [markdown] id="QrpiODrYk3O-"
# ## 게임 진행

# + colab={"base_uri": "https://localhost:8080/"} id="qGcNM7NPjLTp" executionInfo={"status": "ok", "timestamp": 1649746755291, "user_tz": -540, "elapsed": 133893, "user": {"displayName": "Jungi Kim", "userId": "13599710065611566056"}} outputId="a07601ba-ed5f-4ebf-8ee2-3581b6fd3ed7"
p1 = Human_player()
p2 = Random_player()

# 지정된 게임 수를 자동으로 두게 할 것인지 한게임씩 두게 할 것인지 결정
# auto = True : 지정된 판수(games)를 자동으로 진행 
# auto = False : 한판씩 진행
auto = False

# auto 모드의 게임수
games = 100

print("pl player : {}".format(p1.name))
print("p2 player : {}".format(p2.name))

# 각 플레이어의 승리 횟수를 저장
p1_score = 0
p2_score = 0
draw_score = 0

if auto: 
    # 자동 모드 실행
    for j in tqdm(range(games)):
        
        np.random.seed(j)
        env = Environment()
        
        for i in range(10000):
            # p1 과 p2가 번갈아 가면서 게임을 진행
            # p1(1) -> p2(-1) -> p1(1) -> p2(-1) ...
            reward, done = env.move(p1,p2,(-1)**i)
            # 게임 종료 체크
            if done == True:
                if reward == 1:
                    p1_score += 1
                elif reward == -1:
                    p2_score += 1
                else:
                    draw_score += 1
                break

else:                
    # 한 게임씩 진행하는 수동 모드
    np.random.seed(1)
    while True:
        
        env = Environment()
        env.print = False
        for i in range(10000):
            reward, done = env.move(p1,p2,(-1)**i)
            env.print_board()
            if done == True:
                if reward == 1:
                    print("winner is p1({})".format(p1.name))
                    p1_score += 1
                elif reward == -1:
                    print("winner is p2({})".format(p2.name))
                    p2_score += 1
                else:
                    print("draw")
                    draw_score += 1
                break
        
        # 최종 결과 출력        
        print("final result")
        env.print_board()

        # 한게임 더?최종 결과 출력 
        answer = input("More Game? (y/n)")

        if answer == 'n':
            break           

print("p1({}) = {} p2({}) = {} draw = {}".format(p1.name, p1_score,p2.name, p2_score,draw_score))
                
