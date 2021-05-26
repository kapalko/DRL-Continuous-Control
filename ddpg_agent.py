### Derived from Udacity DDPG workbooks and "Continuous Control with Deep Reinforcement Learning"
### https://arxiv.org/pdf/1509.02971.pdf

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update of target parameters
LR_ACTOR = 1e-4         # actor learning rate
LR_CRITIC = 1e-3        # critic learning rate
WEIGHT_DECAY = 0        # L2 weigth decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """The Agent interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): the dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # initialize actor networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) # optimize off local actor gradient
        
        # initialize critic networks
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        
        # initialize random process (N) for action exploration
        # original paper uses Ornstein-Uhlenbeck process with theta = 0.15 and sigma = 0.2
        self.noise = OU_Noise(action_size, random_seed)
        
        # initialize replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def step(self, state, action, reward, next_state, done):
        
        # add state and reward to memory
        self.memory.add(state, action, reward, next_state, done)
        
        # if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        """Takes an action from the policy given our current state and adds noise if requested"""
        state = torch.from_numpy(state).float().to(device)  # converts our observation to a format used by the policy
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using batch of experience tuples.
        y_i = r_i + γ * critic_target(next_state, actor_target(next_state)),
        where y_i are the Q targets, and where:
            actor_target(state) -> action
            critic_target(state, action) -> Q value
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor        
        """
        
        # create variables from experiences tuple
        states, actions, rewards, next_states, dones = experiences
        
        # -----update the critic-----
        # get predicted next-state actions and Q values from targets
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)  # maps the (state, action) pairs to the Q-values
        # compute Q targets for current states, y_i
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))  # we use 1-dones since there won't be any discounted rewards since the episode ends
        # compute the critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # optimize the loss
        self.critic_optimizer.zero_grad()  # always reset the gradient to zero
        critic_loss.backward()
        self.critic_optimizer.step()        
        
        # -----update the actor-----
        # using sampled policy gradient
        # compute the actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # optimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()        
        
        # update the target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def reset(self):
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft updates are used for stability according to the paper
        
        Soft update model parameters.
        θ_target = tau * θ_local + (1 - tau) * θ_target
        
        Params
        ======
            local_model: PyTorch model that weights will be copied from
            target_model: PyTorch model that weights will be copied to
            tau (float): soft update coefficient
        """
        for var, target_var in zip(local_model.parameters(), target_model.parameters()):
            with torch.no_grad():
                target_var.copy(tau * var + (1.0 - tau) * target_var)

class OU_Noise():
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.seed = random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        """Reset the internal state of the process (noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update the internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    

class ReplayBuffer:
    """Buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of the buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to the memory buffer"""
        
        ex = self.experience(state, action, reward, next_state, done)
        self.memory.append(ex)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
        