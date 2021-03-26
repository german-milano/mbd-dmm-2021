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
        # Llamo al constructor del parent
        super(DQNAgent, self).__init__(gym_env, model, obs_processing_func, batch_size, learning_rate, gamma)
        
        # Memoria del agente 
        self.memory = ReplayMemory(memory_buffer_size)

        # Epsilon
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time


    def compute_epsilon(self, steps_so_far):
        # Epsilon linealmente decreciente
        eps = max(self.epsilon_i - steps_so_far * (self.epsilon_i - self.epsilon_f)/self.epsilon_anneal, self.epsilon_f)
        return eps


    def select_action(self, state, current_steps, train=True):
        # Calculo epsilon
        epsilon = self.compute_epsilon(current_steps)
        
        # Obtengo la acción
        action = super(DQNAgent, self).select_action(state, epsilon, train)
            
        return action


    def train(self, number_episodes, max_steps):
        rewards = []                              # Trazabilidad de los rewards por episodio
        episode_steps = np.zeros(number_episodes) # Trazabilidad de steps por episodio
        total_steps = 0

        for ep in tqdm(range(number_episodes), unit=' episodes'):
            # Observar estado inicial como indica el algoritmo
            state = self.state_processing_function(self.env.reset()).to(self.device)
            
            current_episode_reward = 0.0

            for s in range(max_steps):

                # Seleccionar accion usando una política epsilon-greedy.
                action = self.select_action(state, total_steps, train=True)

                # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                current_episode_reward += reward
                episode_steps[ep] += 1
                total_steps += 1

                # Guardar la transicion en la memoria
                self.memory.add(state, action, reward, done, next_state)

                # Actualizar el estado
                state = next_state

                # Actualizar el modelo
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
        if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad() 

            # Obtener un minibatch de la memoria 
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, dones, next_states = list(zip(*batch))

            # Enviar los tensores al dispositivo correspondiente.
            states = (torch.stack(states)).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            dones = torch.tensor(dones).float().to(self.device)
            next_states = (torch.stack(next_states)).to(self.device)

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = self.policy_net(states).gather(dim=1, index=actions.view(-1,1)).squeeze()
  
            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.          
            max_q_next_state = self.policy_net(next_states)
            max_q_next_state = max_q_next_state.max(dim=1)[0]*(1 - dones)
            max_q_next_state = max_q_next_state.detach()

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            target = rewards + self.gamma * max_q_next_state
        
            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            self.loss_function(q_actual, target).backward()
            self.optimizer.step()