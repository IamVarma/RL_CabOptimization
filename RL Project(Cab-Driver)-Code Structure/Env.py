# Import routines

import numpy as np
import math
import random
from itertools import permutations,product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(permutations(range(0,m), 2))
        
        self.state_space = list(product(*[list(range(0,m)), list(range(0,t)), list(range(0,d))]))
    
        self.state_init = random.choice(self.state_space)


        #Variables for On-Hot Encoding
        self.eye_loc=np.eye(5)
        self.eye_hour=np.eye(24)
        self.eye_day=np.eye(7)

        Time_matrix = np.load("TM.npy")
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = list(self.eye_loc[state[0]]) + list(self.eye_hour[state[1]]) + list(self.eye_day[state[2]])

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        requests = 0
        if location == 0:
            requests = np.random.poisson(2)

        if location == 1:
            requests = np.random.poisson(12)
        
        if location == 2:
            requests = np.random.poisson(4)

        if location == 3:
            requests = np.random.poisson(7)

        if location == 4:
            requests = np.random.poisson(8)


        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(0, (m-1)*m ), requests)
         # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0))

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        reward = (R*Time_matrix[action[0]][action[1]][state[1]][state[2]]) - C*(Time_matrix[action[0]][action[1]][state[1]][state[2]] + Time_matrix[state[0]][action[0]][state[1]][state[2]])

        if action == (0,0):
            reward = -C
        
        #print(reward)
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""

        if action == (0,0):
            if state[1] >= 23:
                next_state = (state[0],0 , state[2]+1)
                if state[2] >= 6:
                    next_state = (state[0],0 , 0)
            else: 
                next_state = (state[0], state[1] + 1, state[2])

        else:
            ride_time = Time_matrix[action[0]][action[1]][state[1]][state[2]] + Time_matrix[state[0]][action[0]][state[1]][state[2]]
            #print(ride_time)
            next_hr_day = int((state[1]+ride_time)%24)
            
            next_state = (action[1], next_hr_day, state[2])
            if state[1] == 23:
                next_state = (action[1], next_hr_day, state[2]+1)
                if state[2] >= 6:
                    next_state = (action[1], next_hr_day, 0)
            
          


        return next_state


    def reset(self):
        return self.action_space, self.state_space, self.state_init
