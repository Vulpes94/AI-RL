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

# +
from keras.models import load_model
from Quoridor import Environment,DQN_player

import numpy as np
import copy


# -

class Human_player():
    def __init__(self):
        self.name = "Human_player"
    
    def select_action(self,env, player):
        while True:
            # 가능한 행동을 조사한 후 표시
            available_action = env.get_action(player)
            print("possible actions = {}".format(available_action))
            
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


class Monte_Carlo_player():
    def __init__(self):
        self.name = "MC_player"
        self.num_playout = 1000
    
    def select_action(self,env, player):
        # 가능한 행동을 조사한 후 표시
        available_action = env.get_action(player)
        V = np.zeros(len(available_action))
        
        for i in range(len(available_action)):
            
            # 플레이아웃을 반복
            for j in range(self.num_playout):
                # 현재 상태를 복사해서 플레이아웃에 사용
                temp_env = copy.deepcopy(env)
                
                # p1이 이기면 1, p2가 이기면 -1
                self.playout(temp_env,available_action[i],player)
                if player == temp_env.winner:
                    V[i] += 1
        
        return available_action[np.argmax(V)]
    
    # 플레이아웃 재귀 함수
    # 게임이 종료 상태가 될 때까지 행동을 임의로 선택하는 것을 반복
    def playout(self,temp_env,action,player):
        temp_env.move(player,action)
        temp_env.end_check(player)
        
        # 게임 종료 체크
        if temp_env.done == True:
            return
        else:
            # 플레이어 교체
            if(player ==1):
                player = 2
            else:
                player = 1
                
            # 가능한 행동 조사
            available_action = temp_env.get_action(player)
            
            # 무작위로 행동을 선택
            action = np.random.randint(len(available_action))
            self.playout(temp_env,available_action[action],player)


# +
p1 = DQN_player()
p1.epsilon = 0
p1.load_weights('Q-p1.h5')

# p2 = Monte_Carlo_player()

p2 = Human_player()

while True:
    env = Environment()
    
    # 상단에 플레이어 2, 하단에 플레이어 1 setting
    env.board[0][8] = 2
    env.board[16][8] = 1
    
    for i in range(10000):
        if(i%2==0):
            player = 1
            action = p1.select_action(env, player)
            env.move(player,action)
        else:
            player = 2
            action = p2.select_action(env, player)
            env.move(player ,action)
        
            
        env.end_check(player)
        env.print_board()
        if env.done == True:
            if env.winner == 1:
                print("winner is p1")
            elif env.winner == 2:
                print("winner is p2")
            break
        
    # 최종 결과 출력        
    print("final result")
    env.print_board()

    # 한게임 더?최종 결과 출력 
    answer = input("More Game? (y/n)")

    if answer == 'n':
        break
