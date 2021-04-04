import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

from tqdm.notebook import tqdm

from agents.generic_agent import GenericAgent
from utils.memories import ReplayMemory



class DQNAgent(GenericAgent):

    def __init__(self, gym_env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_time):
        # Calling parent's constructor
        super(DQNAgent, self).__init__(gym_env, model, obs_processing_func, batch_size, learning_rate, gamma)
        
        # Agent's memory
        self.memory = ReplayMemory(memory_buffer_size)

        # Epsilon
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time


    def compute_epsilon(self, steps_so_far):
        '''
        Function to calculate a value which represents epsilon factor of exploration/exploitation

                Parameters:
                        steps_so_far (int): number of steps taken
                        
                Returns:
                        eps (float): calculated value for epsilon factor of exploration/exploitation
        '''
        # Linearly decreasing epsilon
        eps = max(self.epsilon_i - steps_so_far * (self.epsilon_i - self.epsilon_f)/self.epsilon_anneal, self.epsilon_f)
        return eps


    def select_action(self, state, current_steps, train=True):
        '''
        Function to select an action according to its current state

                Parameters:
                        state (tuple): environment state
                        current_steps (int): number of steps taken
                        train (boolean): var to control if current run is for training or not  
                        
                Returns:
                        action (tuple): action to be taken on the environment
        '''
        # Calculate epsilon
        epsilon = self.compute_epsilon(current_steps)
        
        # Get action
        action = super(DQNAgent, self).select_action(state, epsilon, train)
            
        return action


    def train(self, number_episodes, max_steps):
        '''
        Function to train a DQN agent, which returns a list of accumulated rewards per episode and a list
        of the number of steps taken per episode

                Parameters:
                        number_episodes (int): durantion (in episodes) for the training
                        max_steps (int): max number of steps allowed per episode
                        
                Returns:
                        rewards (list): accumulated rewards per episode
                        episode_steps (list): number of steps taken per episode
        '''

        rewards = []                              # Rewards traceability
        episode_steps = np.zeros(number_episodes) # Steps traceability
        total_steps = 0

        for ep in tqdm(range(number_episodes), unit=' episodes'):
            # Observe initial state
            state = self.state_processing_function(self.env.reset()).to(self.device)
            
            current_episode_reward = 0.0

            for s in range(max_steps):

                # Select action using epsilon-greedy policy
                action = self.select_action(state, total_steps, train=True)

                # Execute action, observe results and process them
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                current_episode_reward += reward
                episode_steps[ep] += 1
                total_steps += 1

                # Store transition in replay memory
                self.memory.add(state, action, reward, done, next_state)

                # Update state
                state = next_state

                # Update model
                self.update_weights()

                if done: 
                    break

            rewards.append(current_episode_reward)

            # Report on the traning rewards every 100 episodes
            if ep % 100 == 0:
                print(f"Episode {ep} - Avg. Reward over the last 100 episodes {np.mean(rewards[-100:])}")

        print(f"Episode {ep + 1} - Avg. Reward over the last 100 episodes {np.mean(rewards[-100:])}")

        return rewards, episode_steps


    def update_weights(self):
        '''
        Function to update ANN weights using gradient descent

                Parameters:
                        None
                        
                Returns:
                        None
        '''

        if len(self.memory) > self.batch_size:
            # Reset gradients
            self.optimizer.zero_grad() 

            # Get mini-batch sample from memory
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, dones, next_states = list(zip(*batch))

            # Send tensors to appropiate device
            states = (torch.stack(states)).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            dones = torch.tensor(dones).float().to(self.device)
            next_states = (torch.stack(next_states)).to(self.device)

            # Get Q value for the mini-batch according to policy net
            q_actual = self.policy_net(states).gather(dim=1, index=actions.view(-1,1)).squeeze()
  
            # Get max a' Q value for the next states of the mini-batch
            max_q_next_state = self.policy_net(next_states)
            max_q_next_state = max_q_next_state.max(dim=1)[0]*(1 - dones)
            max_q_next_state = max_q_next_state.detach()

            # Compute DQN's target value
            target = rewards + self.gamma * max_q_next_state
        
            # Compute cost and update weights
            # Should be (predictions, targets)
            self.loss_function(q_actual, target).backward()
            self.optimizer.step()