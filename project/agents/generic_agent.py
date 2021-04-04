import random
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from utils.visualization import show_video


class GenericAgent:
    
    def __init__(self, gym_env, model, obs_processing_func, batch_size, learning_rate, gamma):
        
        # Device to execute ops
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Agent's model
        self.policy_net = model.to(self.device)

        # Cost function (MSE)
        self.loss_function = nn.MSELoss().to(self.device)

        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # State processing function
        self.state_processing_function = obs_processing_func

        # Environment
        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
    
    
    def select_action(self, state, epsilon=0, train=True):
        '''
        Function to select an action according to its current state

                Parameters:
                        state (tuple): environment state
                        epsilon (float): value for epsilon factor of exploration/exploitation
                        train (boolean): var to control if current run is for training or not  
                        
                Returns:
                        action (tuple): action to be taken on the environment
        '''
        
        # Greedy action
        action = torch.argmax(self.policy_net(state))
        action = action.item()
        
        if train:
            # When training, exploration is used
            if np.random.uniform() < epsilon:
                action = np.random.choice(self.env.action_space.n)
            
        return action
    

    def generate_dataset(self, env, action_type='random', epsilon=0.1, num_samples=1000, max_steps=200):
        '''
        Returns a dataset sampled from an environment and its corresponding trajectories

                Parameters:
                        env (gym.Env): selected environment to sample
                        action_type (str): action policy type for environment sampling
                        num_samples (int): number of samples to be taken from the environment
                        
                Returns:
                        dataset (list): list of samples as transitions
                        trajectories (list): list of trajectories (one per episode)
        '''
        
        # Dataset
        dataset = []
        trajectories = []

        while len(dataset) < num_samples:
            # Reset environment
            state = self.state_processing_function(env.reset()).to(self.device)

            # List to save trayectories
            trajectory = []

            done = False

            while not done:
                action = None
                
                if action_type == 'greedy':
                    # Select an action using full-greedy policy
                    action = self.select_action(state, 0, train=False)

                elif action_type == 'e-greedy':
                    # Select an action using epsilon-greedy policy
                    if np.random.uniform() > epsilon:
                        action = self.select_action(state, 0, train=False)
                    else:
                        action = np.random.choice(env.action_space.n)

                else:
                    # Choose random action
                    action = np.random.choice(env.action_space.n)
              
                # Execute action
                next_state, reward, done, _ = env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                # Append tuple to dataset
                dataset.append((state, action, reward, done, next_state))

                # Append tuple to trajectory
                trajectory.append((state, action, reward, done, next_state))

                if done:
                    # Check if trajectory has max_steps transitions
                    # Otherwise, complete trayectory with dummies
                    if len(trajectory) < max_steps:
                        # For MountainCar-v0 env, a dummy is defined as the car reaching the flag
                        # This means:
                        # Position > 0.5
                        # Velocity = 0 (could be any within accepted range)
                        # Action = 1 (don't accelerate)
                        # Reward = 0
                        # Done = True
                        dummy = (torch.Tensor([0.55,0]), 1, 0, True, torch.Tensor([0.55,0]))
                        trajectory.extend(list(repeat(dummy, max_steps-len(trajectory))))
                    
                    # Save trajectory
                    trajectories.append(trajectory)

                    break
                
                # Update state
                state = next_state
        
        return dataset, trajectories


    def record_test_episode(self, env):
        '''
        Function to record (as a video) a test episode

                Parameters:
                        env (gym.Env): selected environment
                        
                Returns:
                        None
        '''

        done = False
    
        # Check initial state
        state = self.state_processing_function(env.reset()).to(self.device)

        while not done:
            env.render()

            # Select action using full-greedy policy 
            action = self.select_action(state, 0, train=False)

            # Execute action, observe results and process them
            next_state, reward, done, _ = env.step(action)
            next_state = self.state_processing_function(next_state).to(self.device)

            if done:
                break      

            # Update state
            state = next_state

        env.close()
        show_video()
