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
        
        # Dispositivo sobre el que correran las ops
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Modelo del agente
        self.policy_net = model.to(self.device)

        # Función de costo (MSE)
        self.loss_function = nn.MSELoss().to(self.device)

        # Optimizador (Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Funcion para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Ambiente
        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
    
    
    def select_action(self, state, epsilon=0, train=True):
        
        # Accion greedy
        action = torch.argmax(self.policy_net(state))
        action = action.item()
        
        if train:
            # Si entreno, exploro
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

        while len(dataset) < num_samples:   # TODO (gmilano): corregir cuando num_samples % max_steps es distinto de cero
            # Reinicio el ambiente 
            state = self.state_processing_function(env.reset()).to(self.device)

            # Trayectoria para guardar las transiciones (por episodio)
            trajectory = []

            done = False

            while not done:
                action = None
                
                if action_type == 'greedy':
                    # Elijo una acción totalmente greedy en función de un agente pre-entrenado
                    action = self.select_action(state, 0, train=False)

                elif action_type == 'e-greedy':
                    # Elijo una acción greedy, manteniendo algo de exploración
                    if np.random.uniform() > epsilon:
                        action = self.select_action(state, 0, train=False)
                    else:
                        action = np.random.choice(env.action_space.n)

                else:
                    # Elijo una acción random
                    action = np.random.choice(env.action_space.n)
              
                # Ejecuto la acción sobre el ambiente
                next_state, reward, done, _ = env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                # Hack porque el ambiente de gym siempre tiene reward -1
                # done = True se da únicamente cuando se llega a la bandera
                #if done:
                #  reward = 0

                # Agrego la tupla al dataset
                dataset.append((state, action, reward, done, next_state))
                # Agrego la tupla a la trayectoria
                trajectory.append((state, action, reward, done, next_state))

                if done:
                    # Reviso que la trayectoria tenga la cantidad de max_steps transiciones
                    # Caso contrario, completo con dummies
                    if len(trajectory) < max_steps:
                        # Para MountainCar-v0, se define el dummy como llegar a la bandera
                        # Position > 0.5
                        # Velocity = 0 (could be any within accepted range)
                        # Action = 1 (Don't accelerate)
                        # Reward = 0
                        # Done = True
                        dummy = (torch.Tensor([0.55,0]), 1, 0, True, torch.Tensor([0.55,0]))
                        trajectory.extend(list(repeat(dummy, max_steps-len(trajectory))))
                    
                    # Guardo la trayectoria
                    trajectories.append(trajectory)

                    break
                
                # Actualizo el estado
                state = next_state
        
        return dataset, trajectories


    def record_test_episode(self, env):
        done = False
    
        # Observar estado inicial como indica el algoritmo
        state = self.state_processing_function(env.reset()).to(self.device)

        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(state, 0, train=False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            next_state, reward, done, _ = env.step(action)
            next_state = self.state_processing_function(next_state).to(self.device)

            if done:
                break      

            # Actualizar el estado
            state = next_state

        env.close()
        show_video()
