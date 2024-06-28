import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.replay_dqn import ReplayBuffer

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class DeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DeepQNetwork, self).__init__()
 
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, action_dim)
 
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
 
    def forward(self, state):
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
 
        q = self.q(x)
 
        return q
 
    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)
 
    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))
 
 
class DQN:
    def __init__(self, state_dim, action_dim, args):
        self.tau = args.tau
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.eps_min = args.eps_end
        self.eps_dec = args.eps_dec
        self.batch_size = args.batch_size
        self.action_space = [i for i in range(action_dim)]
        self.checkpoint_dir = args.ckpt_dir
 
        self.q_eval = DeepQNetwork(alpha=args.lr, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=args.fc1_dim, fc2_dim=args.fc2_dim)
        self.q_target = DeepQNetwork(alpha=args.lr, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=args.fc1_dim, fc2_dim=args.fc2_dim)
 
        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim,
                                   max_size=args.max_size, batch_size=args.batch_size)
 
        self.update_network_parameters(tau=1.0)
 
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
 
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
 
    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)
 
    def select_action(self, observation, isEval=False):
        state = T.FloatTensor([observation]).to(device).unsqueeze(0)
        actions = self.q_eval.forward(state)
        action = T.argmax(actions).item()
        if isEval:
            action = action
        else:
            if (np.random.random() < self.epsilon):
                action = np.random.choice(self.action_space)

        return action
 
    def learn(self):
        if not self.memory.ready():
            return
 
        states, actions, rewards, next_states, terminals = self.memory.sample_buffer()
        batch_idx = np.arange(self.batch_size)
 
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
 
        with T.no_grad():
            q_ = self.q_target.forward(next_states_tensor)
            q_[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * T.max(q_, dim=-1)[0]        # natural DQN
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]
 
        loss = F.mse_loss(q, target.detach())
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
 
        self.update_network_parameters()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
 
    def save_models(self, episode):
        self.q_eval.save_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Saving Q_eval network successfully!')
        self.q_target.save_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Saving Q_target network successfully!')
 
    def load_models(self, episode):
        self.q_eval.load_checkpoint(self.checkpoint_dir + 'Q_eval/DQN_q_eval_{}.pth'.format(episode))
        print('Loading Q_eval network successfully!')
        self.q_target.load_checkpoint(self.checkpoint_dir + 'Q_target/DQN_Q_target_{}.pth'.format(episode))
        print('Loading Q_target network successfully!')
    
    def save(self, seed, filename, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.q_eval.save_checkpoint(directory+'/seed{}_{}_eval.pth'.format(seed, filename))
        self.q_target.save_checkpoint(directory+'/seed{}_{}_target.pth'.format(seed, filename))
    
    def load(self, seed, filename, directory):
        self.q_eval.load_checkpoint(directory+'/seed{}_{}_eval.pth'.format(seed, filename))
        self.q_target.load_checkpoint(directory+'/seed{}_{}_target.pth'.format(seed, filename))

