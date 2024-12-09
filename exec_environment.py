"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import sys
# Change to the path of your ZMQ python API
sys.path.append('/app/zmq/')
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random


class Simulation():
    def __init__(self, sim_port = 23004):
        self.sim_port = sim_port
        self.directions = ['Up','Down','Left','Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')
    
    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break
    
    def getObjectsInBoxHandles(self):
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step
    
    def action(self,direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir*span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

    def calculate_reward(self, positions):
        blue_positions = positions[:9]
        red_positions = positions[9:]

        blue_centroid = np.mean(blue_positions, axis=0)
        red_centroid = np.mean(red_positions, axis=0)
        distance = np.linalg.norm(blue_centroid - red_centroid)

        reward = max(0, 10 - distance)
        return reward


class QNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(128, output_size)


    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class DQNAgent():
   
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(num_states, num_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.num_actions))  # Exploration
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()  # Exploitation

    def updateQNetwork(self, state, action, reward, next_state):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        
        target = q_values.clone() 
        td_error = reward 
        if q_values.numel() > 0: 
            td_error = (self.gamma * torch.max(next_q_values).item() - q_values[action].item())/100 
            target[action] = reward + self.gamma * torch.max(next_q_values).item()


        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error


def isSuccess(env):

    positions = env.getObjectsPositions()
    blue_positions = positions[:9]
    red_positions = positions[9:]

    blue_centroid = np.mean(blue_positions, axis=0)
    red_centroid = np.mean(red_positions, axis=0)
    distance = np.linalg.norm(blue_centroid - red_centroid)

    success_threshold = 1.0
    return distance < success_threshold


def train_agent(env: Simulation, agent: DQNAgent, episodes, steps):
    
    fi = open('training_log.txt', 'w')
    rewards=[]
    for episode in range(episodes):
        print(f'Running episode: {episode + 1}')
        total_reward = 0
        positions = env.getObjectsPositions()
        state = np.array(positions).flatten()

        for _ in range(steps):
            action = agent.choose_action(state)
            env.action(env.directions[action])
            positions = env.getObjectsPositions()
            new_state = np.array(positions).flatten()
            reward = env.calculate_reward(positions)
            td_error = agent.updateQNetwork(state, action, reward, new_state)
            state = new_state
            total_reward += reward

            fi.write(f"TD Error: {td_error}\n")


        rewards.append(total_reward)
        fi.write(f"Reward for episode {episode+1} : {total_reward}\n")
        

    # Saving Q-network weights in a text file
    with open("q_network_weights.txt", 'w') as f:
        for param_tensor, weights in agent.q_network.state_dict().items():
            f.write(f"{param_tensor}:\n")
            f.write(f"{weights}\n")


def test_agent(env: Simulation, agent: DQNAgent, episodes):

    results=[]
    success_count=0
    total_time_taken=0
    for episode in range(episodes):
        start_time = time.time()
        state = np.array(env.getObjectsPositions()).flatten()
        total_reward = 0

        print(f'Running episode: {episode + 1}')
        for step in range(5):
            action = agent.choose_action(state)
            env.action(env.directions[action])
            reward = env.calculate_reward(env.getObjectsPositions())
            total_reward += reward
        
        end_time = time.time()
        episode_time = end_time - start_time
        total_time_taken += episode_time

        success = isSuccess(env)
        if bool(success):
            success_count+=1
        results.append((bool(success), float(total_reward), float(episode_time)))
    
    return results, total_time_taken, success_count


def main():

    num_states = 36
    num_actions = 4
    alpha = 0.01
    gamma = 0.99
    epsilon = 0.3

    env = Simulation()
    agent = DQNAgent(num_states, num_actions, alpha, gamma, epsilon)
    train_agent(env, agent, episodes=20, steps=20)
    env.stopSim()

    env = Simulation()
    test_results_dq_learning, DQ_time, success_count = test_agent(env, agent, episodes=100)

    with open('test_results.txt', 'w') as f:
        f.write("Results based on DQ-Learning Action:\n")
        for k,result in enumerate(test_results_dq_learning):
            f.write(f"Reward for episode {k+1} : {result[1]}\n")
            if result[0]:
                f.write(f"Sucessful mix  Time Taken (in Sec): {result[2]}\n")
            else:
                f.write("Unsuccesful mix  Time Taken (in Sec): {result[2]}\n")
        f.write(f"\nNumber of Successful attempts: {success_count}")
        f.write(f"\nTotal time taken : {DQ_time} seconds")

    env.stopSim()


if __name__ == '__main__':
    
    main()
