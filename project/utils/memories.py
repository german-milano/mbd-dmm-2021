from collections import namedtuple

import random



Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

class GenericMemory:
    
    def __init__(self, data=None):
        if data is None:
            self.memory = []
        else:
            self.memory = data

            
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    
    def __len__(self):
        return len(self.memory)
    

class ReplayMemory(GenericMemory):

    def __init__(self, buffer_size):
        super(ReplayMemory, self).__init__()
        self.buffer_size = buffer_size
        self.position = 0

        
    def add(self, state, action, reward, done, next_state):
        # Implementar.
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        newElement= Transition(state, action, reward, done, next_state)
        self.memory[self.position] = newElement
        self.position = (self.position + 1) % self.buffer_size
