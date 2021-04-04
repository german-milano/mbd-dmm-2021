from collections import namedtuple

import random


# Define tuple for transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

class GenericMemory:
    
    def __init__(self, data=None):

        # Initialize memory
        if data is None:
            self.memory = []
        else:
            self.memory = data

            
    def sample(self, batch_size):
        '''
        Function to sample transitions from memory

                Parameters:
                        batch_size (int): size (in transitions) for batch sampling
                        
                Returns:
                        sample of tuples
        '''
        return random.sample(self.memory, batch_size)

    
    def __len__(self):
        '''
        Function to check the length of the memory

                Parameters:
                        None
                        
                Returns:
                        size of memory
        '''
        return len(self.memory)
    

class ReplayMemory(GenericMemory):

    def __init__(self, buffer_size):
        # Calling parent's constructor
        super(ReplayMemory, self).__init__()
        
        # Buffer size for the circular memory
        self.buffer_size = buffer_size
        
        # Current position
        self.position = 0

        
    def add(self, state, action, reward, done, next_state):
        '''
        Function to add a new transition to the replay memory (using FIFO strategy)

                Parameters:
                        state (tensor): state s_t
                        action (tensor): action a_t
                        reward (tensor): reward r_{t+1}
                        done (tensor): ctrl var to check if episode should be terminated
                        next_state (tensor): state s_{t+1}
                        
                Returns:
                        None
        '''

        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        
        # Create new element
        newElement= Transition(state, action, reward, done, next_state)

        # Add new element at current position
        self.memory[self.position] = newElement
        
        # Update position
        self.position = (self.position + 1) % self.buffer_size
