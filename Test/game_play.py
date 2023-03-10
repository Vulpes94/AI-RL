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

from keras.models import load_model
from Quoridor import Environment,DQN_player


class Human_player():
    def __init__(self):
        self.wallCount = 10
    
    def policy(self,env, player):
        while True:
            # 가능한 행동을 조사한 후 표시
            available_action = env.get_action(player,self.wallCount)
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


# +
p2 = DQN_player()
p2.epsilon = 0
p2.load_weights('.h5')

p1 = Human_player()

while True:
    env = Environment()
    
    # 상단에 플레이어 2, 하단에 플레이어 1 setting
    env.board[0][8] = 2
    env.board[16][8] = 1
    
    # 벽 사용 가능 수 다시 복원
    p2.wallCount = 10
    p1.wallCount = 10
    
    for i in range(10000):
        if(i%2==0):
            player = 2
            position = p2.policy(env, player)
            env.move(player ,position, p2)
        else:
            player = 1
            position = p1.policy(env, player)
            env.move(player,position,p1)
        
            
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
