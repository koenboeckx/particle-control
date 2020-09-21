
from random import triangular
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from collections import namedtuple
import random
import argparse
import pickle

Experience = namedtuple('Experience', ['obs', 'action', 'next_obs'])

class ExperienceBuffer:
    def __init__(self):
        self._buffer = []
    def append(self, item):
        self._buffer.append(item)
    def sample(self, k):
        return random.sample(self._buffer, k)
    def __len__(self):
        return len(self._buffer)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4+2, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc3 = nn.Linear(args.n_hidden, args.n_hidden)
        self.out = nn.Linear(args.n_hidden, 4)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
    
    def forward(self, obs, actions):
        x = torch.cat((torch.tensor(obs).float().to(args.device),
                       torch.tensor(actions).float().to(args.device)), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
    
    def save(self, dest_file):
        with open(dest_file, 'wb') as dst:
            torch.save(self.state_dict(), dst)
    
    def load(self, src_file):
        with open(src_file, 'rb') as src:
            self.load_state_dict(torch.load(src, map_location=args.device))
            

def parse_commandline():
    parser = argparse.ArgumentParser(description="Train a model of the multiagent-particle agent")
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Size of the experience buffer')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size')
    parser.add_argument('--n_iters', type=int, default=50000, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning')
    parser.add_argument('--print_interval', type=int, default=100, help='How often to print training loss')
    parser.add_argument('--n_hidden', type=int, default=128, help='Size of the hidden layer')
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
        print("Using cuda")
    else:
        args.device = torch.device("cpu")
        print("Using cpu")
    return args

def create_buffer(env, size):
    buffer = ExperienceBuffer()
    validation_set = ExperienceBuffer() 
    obs_n = env.reset()
    obs = obs_n[0]
    for _ in range(size):
        action = np.random.randn(2)
        act_n = [(0, action[0], 0, action[1], 0)]
        obs_n, reward_n, done_n, _ = env.step(act_n)
        next_obs = obs_n[0]
        buffer.append(Experience(obs, action, next_obs))
        next_obs = obs[:]
    for _ in range(size):
        action = np.random.randn(2)
        act_n = [(0, action[0], 0, action[1], 0)]
        obs_n, reward_n, done_n, _ = env.step(act_n)
        next_obs = obs_n[0]
        validation_set.append(Experience(obs, action, next_obs))
        next_obs = obs[:]
    return buffer, validation_set

def train(net, buffer, eval_set, args):
    net.train()
    eval_loss = None
    for i in range(args.n_iters):
        batch = buffer.sample(args.batch_size)
        obs, actions, next_obs  = zip(*batch)
        predictions = net(obs, actions)
        loss = F.mse_loss(torch.tensor(next_obs).to(args.device), predictions)
        if i % args.print_interval == 0:
            if eval_loss is None:
                eval_loss = eval(net, eval_set.sample(args.batch_size), args)
            else:
                eval_loss = 0.9 * eval_loss + 0.1 * eval(net, eval_set.sample(args.batch_size), args)
            print(f"Step {i} - loss = {eval_loss:8.5f}")

        net.optimizer.zero_grad()
        loss.backward()
        net.optimizer.step()
    return net

def eval(net, eval_set, args):
    net.eval()
    obs, actions, next_obs  = zip(*eval_set)
    with torch.no_grad():
        prediction = net(obs, actions)
        loss = F.mse_loss(torch.tensor(next_obs).to(args.device), prediction).item()
    net.train()
    return loss

if __name__ == '__main__':
    args = parse_commandline()
    print(args)
    scenario = scenarios.load(args.scenario).Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    buffer, eval_set =   create_buffer(env, size=args.buffer_size)

    with open('buffer.pkl', 'wb') as dst:
        pickle.dump([buffer, eval_set], dst)
    with open('buffer.pkl', 'rb') as src:
        buffer, eval_set = pickle.load(src)
    print(f'Buffer created with size {len(buffer)}')
    net = Network()
    net.to(args.device)
    # net.load('model.torch')
    net = train(net, buffer, eval_set, args)
    net.save('model.torch')

    print(net)