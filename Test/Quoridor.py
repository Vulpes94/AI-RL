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

# # Quoridor AI

# ## 필요한 라이브러리 임포트

# +
import numpy as np
from tqdm import tqdm
import copy

from keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from keras import metrics
from keras.layers import Dense, Flatten, Conv2D

from path import astar


# -

# ## Environment

class Environment():
    def __init__(self):
    # 보드는 0으로 초기화된 17x17개의 배열 준비
    # 게임종료 : done = True
        self.board = [[0 for j in range(17)] for i in range(17)]
        self.done = False
        self.reward = 0
        self.winner = 0
        self.wall = -1
        self.print = False
        self.player1wallcount = 10
        self.player2wallcount = 10
        
    def move(self,player,action):
        pos = [[i,j] for i in range(17) for j in range(17) if self.board[i][j]==player]
        
        # 보드에 플레이어의 선택을 표시
        if(action == 0):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] -2][pos[0][1]] = player
        elif(action == 1):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] -2][pos[0][1] +2] = player
        elif(action == 2):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0]][pos[0][1] + 2] = player
        elif(action == 3):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] + 2][pos[0][1] + 2] = player
        elif(action == 4):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] +2][pos[0][1]] = player
        elif(action == 5):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] + 2][pos[0][1] - 2] = player            
        elif(action == 6):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0]][pos[0][1] - 2] = player
        elif(action == 7):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] - 2][pos[0][1] - 2] = player
        elif(action == 8):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] -4][pos[0][1]] = player
        elif(action == 9):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0]][pos[0][1] + 4] = player
        elif(action == 10):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0] +4][pos[0][1]] = player
        elif(action == 11):
            self.board[pos[0][0]][pos[0][1]] = 0
            self.board[pos[0][0]][pos[0][1] -4] = player
        elif(action >= 12 and action <= 75):
            if player==1: 
                self.player1wallcount += -1
            else: 
                self.player2wallcount += -1
                
            if((action-11)%8 != 0):
                x = int(2 * ((action-11)%8) - 2)
                y = int(2 * ((action-11)//8) + 1)
                self.board[x][y] = self.wall
                self.board[x+1][y] = self.wall
                self.board[x+2][y] = self.wall
                if self.print:
                    print("x : ",x+1,"y : ",y)
            else:
                x = 14
                y = int(2 * ((action-11)//8) -1)
                self.board[x][y] = self.wall
                self.board[x+1][y] = self.wall
                self.board[x+2][y] = self.wall
                if self.print:
                    print("x : ",x+1,"y : ",y)
        elif(action >= 76 and action <= 139):
            if player==1: 
                self.player1wallcount += -1
            else: 
                self.player2wallcount += -1
                
            if((action-75)%8 != 0):
                x = int(2 * ((action-75)//8) + 1)
                y = int(2 * ((action-75)%8) - 2)
                self.board[x][y] = self.wall
                self.board[x][y+1] = self.wall
                self.board[x][y+2] = self.wall
                if self.print:
                    print("x : ",x,"y : ",y+1)
            else:
                x = int(2 * ((action-75)//8) -1)
                y = 14
                self.board[x][y] = self.wall
                self.board[x][y+1] = self.wall
                self.board[x][y+2] = self.wall
                if self.print:
                    print("x : ",x,"y : ",y+1)
        
        if self.print:
            print("player : ",player,"action : ",action)
            self.print_board()
        
    # 현재 보드 상태에서 가능한 행동(최대 140)을 탐색하고 리스트로 반환
    def get_action(self,player):
        observation = []
        pos = [[i,j] for i in range(17) for j in range(17) if self.board[i][j]==player]
        
        east = pos[0][1] + 2; eeast = pos[0][1] + 4;
        west = pos[0][1] - 2; wwest = pos[0][1] - 4;
        north = pos[0][0] - 2; nnorth = pos[0][0] - 4;
        south =  pos[0][0]+ 2; ssouth =  pos[0][0] + 4;
        
        N = 0 ;NE = 1 ;E = 2 ;SE = 3
        S = 4 ;SW = 5 ;W = 6 ;NW = 7
        NN = 8;EE = 9;SS=10;WW=11
        
        # 동쪽 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
        if (east <= 16 and self.board[pos[0][0]][east - 1] != self.wall):
            if (self.board[pos[0][0]][east] == 0):
                observation.append(E)
            # 동쪽에 플레이어가 있을 때
            else:
                # 동쪽 두칸 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽에 없을 때)
                if (eeast <= 16 and self.board[pos[0][0]][eeast - 1] != self.wall):
                    observation.append(EE)
                else:
                    # 대각선(북동) 이동 (도착지의 북쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (north >= 0 and self.board[north + 1][east] != self.wall):
                      observation.append(NE)
                    
                    # 대각선(남동) 이동 (도착지의 남쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (south <= 16 and self.board[south - 1][east] != self.wall):
                      observation.append(SE)
                
        # 서쪽 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
        if (west >= 0 and self.board[pos[0][0]][west + 1] != self.wall):
            if (self.board[pos[0][0]][west] == 0):
                observation.append(W)
            # 서쪽에 플레이어가 있을 때
            else:
                # 서쪽 두칸 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽에 없을 때)
                if (wwest >= 0 and self.board[pos[0][0]][wwest + 1] != self.wall):
                    observation.append(WW)
                else:
                    # 대각선(북서) 이동 (도착지의 북쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (north >= 0 and self.board[north + 1][west] != self.wall):
                      observation.append(NW)
                    
                    # 대각선(남서) 이동 (도착지의 남쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (south <= 16 and self.board[south - 1][west] != self.wall):
                      observation.append(SW)     

        # 북쪽 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
        if (north >= 0 and self.board[north + 1][pos[0][1]] != self.wall):
            if (self.board[north][pos[0][1]] == 0):
                observation.append(N)
            # 북쪽에 플레이어가 있을 때
            else:
                # 북쪽 두칸 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽에 없을 때)
                if (nnorth >= 0 and self.board[nnorth + 1][pos[0][1]] != self.wall):
                    observation.append(NN)
                else:
                    # 대각선(북서) 이동 (도착지의 서쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (west >= 0 and self.board[north][west + 1] != self.wall):
                        if NW not in observation:
                          observation.append(NW)
                    
                    # 대각선(북동) 이동 (도착지의 동쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (east <= 16 and self.board[north][east - 1] != self.wall):
                        if NE not in observation:
                            observation.append(NE)

        # 남쪽 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
        if (south <= 16 and self.board[south - 1][pos[0][1]] != self.wall):
            if (self.board[south][pos[0][1]] == 0):
                observation.append(S)
            # 남쪽에 플레이어가 있을 때
            else:
                # 남쪽 두칸 이동 (도착지가 보드를 이탈하지 않고 이동경로에 벽에 없을 때)
                if (ssouth <= 16 and self.board[ssouth - 1][pos[0][1]] != self.wall):
                    observation.append(SS)
                else:
                    # 대각선(남서) 이동 (도착지의 서쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (west >= 0 and self.board[south][west + 1] != self.wall):
                        if SW not in observation:
                            observation.append(SW)
                    
                    # 대각선(남동) 이동 (도착지의 동쪽이 보드를 이탈하지 않고 이동경로에 벽이 없을 때)
                    if (east <= 16 and self.board[south][east - 1] != self.wall):
                        if SE not in observation:
                            observation.append(SE)
        
        wallcount = self.player1wallcount if player==1 else self.player2wallcount
        if(wallcount > 0):
            for i in range(1,len(self.board),2):
                for j in range(1,len(self.board[len(self.board)-1]),2):
                    if(self.board[i][j] == 0):
                        if(self.board[i-1][j] == 0 and self.board[i+1][j] == 0):
                            if (self.astarTest((i-1),j)):
                                x = (i-1)/2 + 1 + 8* ((j-1)/2) +11
                                observation.append(int(x)) 
                        if(self.board[i][j-1] == 0 and self.board[i][j+1] == 0):
                            if(self.astarTest(i,(j-1))):
                                x = (j-1)/2 + 1 + 8* ((i-1)/2) +64 +11
                                observation.append(int(x))
        
        return sorted(observation)
    
    def astarTest(self,i,j):
        mat = copy.deepcopy(self.board)
        
        # mat에 벽 설치 해보기
        if(i%2==0 and j%2==1):
            mat[i][j] = -1 ;mat[i+1][j] = -1; mat[i+2][j] = -1
        elif(i%2==1 and j%2==0):
            mat[i][j] = -1 ;mat[i][j+1] = -1; mat[i][j+2] = -1
            
        # 벽 2개 나란히 세웠을 때 틈새 막기
        for i in range(1,len(mat),2):
            for j in range(1,len(mat[len(mat)-1]),2):
                if ((mat[i][j - 1] == self.wall and mat[i][j + 1] == self.wall) or 
                    (mat[i - 1][j] == self.wall and mat[i + 1][j] == self.wall)):
                  mat[i][j] = self.wall

        p1_pos = [[i,j] for i in range(17) for j in range(17) if self.board[i][j]==1]
        p1_start = p1_pos[0]
        s_path_p1 = False
        
        p2_pos = [[i,j] for i in range(17) for j in range(17) if self.board[i][j]==2]
        p2_start = p2_pos[0]
        s_path_p2 = False
        
        # mat에 표시되어 있는 플레이어 제거
        mat[p1_pos[0][0]][p1_pos[0][1]] = 0
        mat[p2_pos[0][0]][p2_pos[0][1]] = 0
        
        end_array = [0,2,4,6,8,10,12,14,16]


        for i in range(len(end_array)):
            path = astar(mat, p1_start, (0,end_array[i]))
            if (path != None):
                s_path_p1 = True
                break

        for i in range(len(end_array)):
            path = astar(mat, p2_start, (16,end_array[i]))
            if (path != None):
                s_path_p2 = True
                break
                
        if (s_path_p1 == True and s_path_p2 == True):
            # 길을 막지 않음
            return True
        else: 
            # 길을 막음
            return False
        
    def end_check(self,player):
        # 상단에 플레이어 2, 하단에 플레이어 1 setting
        end_condition = [0,2,4,6,8,10,12,14,16]
        for line in end_condition:
            if((player == 1 and self.board[0][line] == 1) or player == 2 and self.board[16][line] == 2):
                self.done = True
                self.winner = player
                return
    
    def print_board(self):
        for i in self.board:
            for j in i:
                print(j, end=' ')
            print()


# ## DQN_Player

class DQN_player():
    def __init__(self):
        self.name = "DQN_player"
        self.epsilon = 1
        self.learning_rate = 0.1
        self.gamma=0.9
        
        # 두개의 신경망을 생성
        self.main_network = self.make_network()
        self.target_network = self.make_network()
        # 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
        self.copy_network()
        
        self.print = False

    
    # 신경망 생성
    def make_network(self):
        self.model = Sequential()
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu', input_shape=(17,17,3)))
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (17, 17), padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='tanh'))
        self.model.add(Dense(512, activation='tanh'))
        self.model.add(Dense(256, activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(140))
        
        self.model.compile(optimizer = SGD(learning_rate=0.01), loss = 'mean_squared_error', metrics=['mse'])
        
        return self.model

    # 신경망 복사
    def copy_network(self):
         self.target_network.set_weights(self.main_network.get_weights())
    
    def load_weights(self,model):
        self.model.load_weights(model)

    def save_network(self,name):
        filename = name + '.h5'
        self.main_network.save(filename)
        print("end save model")
        
    def state_convert(self, board):
        # 1은 p1 ,2는 p2 ,-1은 wall
        d_state = np.full((17,17,3),0.1)
        for i in range(17):
            for j in range(17):
                if board[i][j] == 1:
                    d_state[i,j,0] = 1
                elif board[i][j] == 2:
                    d_state[i,j,1] = 1
                elif board[i][j] == -1:
                    d_state[i,j,2] = 1
                else:
                    pass
        return d_state
        
    def select_action(self, env, player):
        action = self.policy(env, player)

        return action
    
    def policy(self, env, player):
        if self.print:
            print("----------- policy start -------------")
        
        # 행동 가능한 상태를 저장
        available_state = env.get_action(player)

        state_3d = self.state_convert(env.board)
        x = np.array([state_3d],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]
        
        if self.print:
            print("{} : policy state".format(available_state))
            print("{} : qvalues".format(np.round(qvalues,3)))
        
        # 행동 가능한 상태의 Q-value를 저장
        available_state_qvalues = qvalues[available_state]
        
        if self.print:
            print("{} : available_state_qvalues".format(np.round(available_state_qvalues,3)))
        
        # max Q-value를 탐색한 후 저장
        greedy_action = np.argmax(available_state_qvalues)
        if self.print:
            print("{} : self.epsilon".format(self.epsilon))
            print("{} : greedy_action".format(greedy_action))
            print("{} : qvalue".format(available_state_qvalues[greedy_action]))
        
        # max Q-value와 같은 값이 여러개 있는지 확인한 후 double_check에 상태를 저장
        double_check = (np.where(qvalues == np.max(available_state[greedy_action]),1,0))
        
        #  여러개 있다면 중복된 상태중에서 다시 무작위로 선택    
        if np.sum(double_check) > 1:
            if self.print:
                print("{} : double_check".format(np.round(double_check,2)))
            double_check = double_check/np.sum(double_check)
            greedy_action =  np.random.choice(range(0,len(double_check)), p=double_check)
            if self.print:
                print("{} : greedy_action".format(greedy_action))
                print("{} : double_check".format(np.round(double_check,2)))
                print("{} : selected state".format(available_state[greedy_action]))
        
        # ε-greedy
        pr = np.zeros(len(available_state))
        
        for i in range(len(available_state)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))
            else:
                pr[i] = self.epsilon / len(available_state)
                if pr[i] < 0:
                    print("{} : - pr".format(np.round(pr[i],2)))

        action = np.random.choice(range(0,len(available_state)), p=pr)
        
        if self.print:
            print("{} : pr".format(np.round(pr,2)))
            print("{} : state[action]".format(available_state[action]))
            print("-----------   policy end -------------")
            
        return available_state[action]
        
        
    def learn_dqn(self,board_backup, action_backup, env, reward, player):
        # 입력을 3차원으로 변환한 후, 메인 신경망으로 q-value를 계산
        new_state = self.state_convert(board_backup)
        x = np.array([new_state],dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]
        before_action_value = copy.deepcopy(qvalues)
        delta = 0
        
        if self.print:
            print("-----------   learn_qtable start -------------")
            print("{} : board_backup".format(board_backup))
            print("{} : action_backup".format(action_backup))
            print("{} : reward.".format(reward))
    
        if env.done == True:
            if self.print:
                print("{} : delta".format(delta))
                print("{} : before update : actions[action_backup]".format(np.round(qvalues[action_backup],3)))
            
            # 게임이 좀료됐을때 신경망의 학습을 위한 정답 데이터를 생성
            qvalues[action_backup] = reward
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            # 생성된 정답 데이터로 메인 신경망을 학습
            self.main_network.fit(x, y, epochs=10, verbose=2)
            
            if self.print:
                after_action_value = copy.deepcopy(self.main_network.predict(x)[0,:])
                delta = after_action_value - before_action_value
                print("{} : before_action_value id = {}".format(np.round(before_action_value,3),id(before_action_value)))
                print("{} : after_action_value id = {}".format(np.round(after_action_value,3),id(after_action_value)))
                print("{} : delta action value".format(np.round(delta,3)))
                state = [[0 for j in range(17)] for i in range(17)]
                state_3d = self.state_convert(state)
                x = np.array([state_3d],dtype=np.float32).astype(np.float32)
                qvalues = self.main_network.predict(x)[0,:]
                print("{} : initial state qvalues".format(np.round(qvalues,3)))

        else:
            # 게임이 진행중일때  신경망의 학습을 위한 정답 데이터를 생성
            # 현재 상태에서 최고 Q 값을 계산
            new_state = self.state_convert(env.board)
            next_x = np.array([new_state],dtype=np.float32).astype(np.float32)
            next_qvalues = self.target_network.predict(next_x)[0,:]
            available_state = env.get_action(player)
            maxQ = np.max(next_qvalues[available_state])     
            
            if self.print:
                print("{} : old_qvalue".format(np.round(before_action_value[action_backup],3)))
                print("{} : next_qvalue".format(np.round(next_qvalues,3)))
                print("{} : available_state".format(np.round(available_state,3)))
                print("{} : maxQ".format(np.round(maxQ,3)))
            
            delta = self.learning_rate*(reward + self.gamma * maxQ - qvalues[action_backup])
            
            if self.print:
                print("{} : delta".format(np.round(delta,3)))
                print("{} : before_update_qvalues".format(np.round(qvalues,3)))
                print("{} : before_update_qvalue".format(np.round(qvalues[action_backup],3)))
                
            qvalues[action_backup] += delta
            
            if self.print:
                print("{} : after_update_qvalue".format(np.round(qvalues[action_backup],3)))            
                print("{} : before_update_qvalues".format(np.round(qvalues,3)))
                target_action_value = copy.deepcopy(qvalues)
                print("{} : new_qvalues".format(np.round(qvalues,3)))            
                print("{} : target_action_value id = {}".format(np.round(target_action_value,3),target_action_value))            
            # 생성된 정답 데이터로 메인 신경망을 학습
            y=np.array([qvalues],dtype=np.float32).astype(np.float32)
            self.main_network.fit(x, y, epochs = 10, verbose=0)
            
            if self.print:
                after_action_value = copy.deepcopy(self.main_network.predict(x)[0,:])
                delta = after_action_value - before_action_value
                print("{} : before_action_value id = {}".format(np.round(before_action_value,3),id(before_action_value)))
                print("{} : target_action_value id = {}".format(np.round(target_action_value,3),id(target_action_value)))
                print("{} : after_action_value id = {}".format(np.round(after_action_value,3),id(after_action_value)))
                print("{} : delta action value".format(np.round(delta,3)))
                state = [[0 for j in range(17)] for i in range(17)]
                state_3d = self.state_convert(state)
                x = np.array([state_3d],dtype=np.float32).astype(np.float32)
                qvalues = self.main_network.predict(x)[0,:]
                print("{} : initial state qvalues".format(np.round(qvalues,3)))

            
        if self.print:
            print("-----------   learn_qtable end -------------")


# ## Train

# +
def main():
    p1_DQN = DQN_player()
    p2_DQN = DQN_player()

    p1_DQN.print = False
    p2_DQN.print = False

    p1_score = 0
    p2_score = 0

    max_learn = 20

    print("p1 player is {}".format(p1_DQN.name))
    print("p2 player is {}".format(p2_DQN.name))

    # p1_DQN.load_weights('./Q-P1-10.h5')
    # p2_DQN.load_weights('./Q-P2-10.h5')

    for j in tqdm(range(max_learn)):
        np.random.seed(j)
        env = Environment()
        env.print = False
        # 상단에 플레이어 2, 하단에 플레이어 1 setting
        env.board[0][8] = 2
        env.board[16][8] = 1


        # 시작할 때 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
        p1_DQN.epsilon = 0.7
        p1_DQN.copy_network()

        p2_DQN.epsilon = 0.7
        p2_DQN.copy_network()

        for i in range(10000):
            # p1 행동을 선택
            player = 1
            action = p1_DQN.policy(env, player)

            p1_board_backup = tuple(env.board)
            p1_action_backup = action

            env.move(player,action)
            env.end_check(player)

            # 게임 종료라면
            if env.done == True:
                # p1의 승리이므로 마지막 행동에 보상 +1
                # p2는 마지막 행동에 보상 -1
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 1,1)
                p2_DQN.learn_dqn(p2_board_backup, p2_action_backup, env, -1,2)
                p1_score += 1
                break

            # p2 행동을 선택
            player = 2
            action = p2_DQN.policy(env,player)

            p2_board_backup = tuple(env.board)
            p2_action_backup = action

            env.move(player,action)
            env.end_check(player)

            if env.done == True:
                # p2승리 = p1 패배 마지막 행동에 보상 -1
                # 지면 보상 : -1
                p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, -1,1)
                p2_DQN.learn_dqn(p2_board_backup, p2_action_backup, env, 1,2)
                p2_score += 1
                break

            # 게임이 끝나지 않았다면 p1의 Q-talble 학습
            p1_DQN.learn_dqn(p1_board_backup, p1_action_backup, env, 0,1)
            p2_DQN.learn_dqn(p2_board_backup, p2_action_backup, env, 0,2)

        # 5게임마다 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
        if j%5 == 0:
            p1_DQN.copy_network()
            p2_DQN.copy_network()
            p1_DQN.save_network("Q-P1-" + str(j))
            p2_DQN.save_network("Q-P2-" + str(j))

    print("p1 = {} p2 = {}".format(p1_score, p2_score))
    print("end learn")
# -

if __name__ == '__main__':
    main()
