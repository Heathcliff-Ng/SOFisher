"""
Created on Feb 21, 2024
@author: Zhuo Li

The main script of trainning and evaluating the SAC algorithm for AD Seeking
Without state normalization
"""
import time
import random
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

# from env.env_cortex import SpatOmics_dis
from env.env_cortex_2cell import SpatOmics_dis
from src.dqn import DQN
from src.args import get_dqn_args


class Runner:
    def __init__(self, args, seed, exps):
        self.args = args
        self.seed = seed
        self.exps = exps
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.grid_x, self.grid_y = self.args.grid_x, self.args.grid_y
        # observation and action spaces
        n_obs = self.grid_x*self.grid_y*self.args.cell_num + 2 + 17 
        self.observation_space = spaces.Box(low = -np.array(np.ones(n_obs)), high = np.array(np.ones(n_obs)), dtype=np.float32)
        self.action_space = spaces.Discrete(24)
        self.args.obs_dim = self.observation_space.shape[0]  # obs dimensions
        self.args.action_dim = self.action_space.n  # actions dimensions
        # Create an agent
        self.agent = DQN(self.args.obs_dim, self.args.action_dim, args)


        self.writer = SummaryWriter(log_dir='runs/seed_{}'.format(self.seed)) # Create a tensorboard
        self.train_rewards = [] # Record the rewards during the training
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.best_eval_rew = - float("Inf") # Initialize the best reward

        print("exp_num: {}, cell_num: {}".format(len(exps), self.args.cell_num))

    def make_folders(self, ):
        data_folder = 'data'
        model_folder = 'model'
        data_seed_folder = os.path.join(data_folder, f'seed_{self.seed}')
        self.model_seed_folder = os.path.join(model_folder, f'seed_{self.seed}')
        self.data_train_folder = os.path.join(data_seed_folder, 'train')
        self.eval_folder = os.path.join(data_seed_folder, 'eval')


        if not os.path.exists(self.data_train_folder):
            os.makedirs(self.data_train_folder)
        if not os.path.exists(self.model_seed_folder):
            os.makedirs(self.model_seed_folder)
        if not os.path.exists(self.eval_folder):
            os.makedirs(self.eval_folder)

    def run(self):
        epi_num = 0
        while self.total_steps < self.args.max_steps:
            # Create env
            exp = random.choice(self.exps)
            env = SpatOmics_dis(self.args, exp)
            state = env.reset()
            epi_num += 1
            epi_steps = 0
            episode_reward = 0
            time_episode_start = time.time()
            for _ in range(self.args.episode_size): # Each episode (1000 steps)
                action = self.agent.select_action(state)
                next_state, reward, done  = env.step(action)  # Step
                self.agent.remember(state, action, reward, next_state, done) # memory push
                self.agent.learn() #including update parameters
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                epi_steps += 1
                if self.total_steps % self.args.eval_freq == 0:
                   self.evaluate_policy()   
                if done:
                    break
            time_episode_end = time.time()
            print("Training: Episode:{}, Total steps: {}, epi steps: {},epi rewards: {}, epi time: {}".
                  format(epi_num, self.total_steps, epi_steps, round(episode_reward,2), round((time_episode_end - time_episode_start),2)))

            self.writer.add_scalar('train_rewards', episode_reward, global_step=self.total_steps)
            self.train_rewards.append(episode_reward)
            np.save('./{}/train_episode_reward.npy'.format(self.data_train_folder), self.train_rewards)



    def evaluate_policy(self, ):
        # Create env
        exp = random.choice(self.exps)
        env_evaluate = SpatOmics_dis(self.args, exp)
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            state = env_evaluate.reset()
            episode_reward = 0
            obses = []
            AD_countses= []
            for _ in range(self.args.episode_size):
                action = self.agent.select_action(state, isEval=True)
                next_state, reward, done = env_evaluate.step(action)  # Step
                state = next_state
                episode_reward += reward
                AD_countses.append(episode_reward)
                obses.append(state[0:2])
                if done:
                    break
            evaluate_reward += episode_reward
        
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)

        print("========================================")
        print("Evaluation: Total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        print("========================================")

        np.save('./{}/evaluate_episode_rewards.npy'.format(self.eval_folder), self.evaluate_rewards)
        self.writer.add_scalar('evaluate_rewards', evaluate_reward, global_step=self.total_steps)

        # Save the models
        self.agent.save(self.seed, "final", self.model_seed_folder)
        # Save the best model
        if self.evaluate_rewards[-1] > self.best_eval_rew:
            self.best_eval_rew = self.evaluate_rewards[-1]
            self.agent.save(self.seed, "best", self.model_seed_folder) 


if __name__ == '__main__':
    args = get_dqn_args()
    print("Start running {} for Sampling in Spatial Omics".format(args.agent_name))
    exps = ['mouse1_slice1', 'mouse1_slice10','mouse1_slice21', 'mouse1_slice31', 
            'mouse1_slice40', 'mouse1_slice50', 'mouse1_slice62', 'mouse1_slice71',
            'mouse1_slice81', 'mouse1_slice91', 'mouse1_slice102', 'mouse1_slice112', 
            'mouse1_slice122', 'mouse1_slice131', 'mouse1_slice153', 'mouse1_slice162',
            'mouse1_slice170', 'mouse1_slice190', 'mouse1_slice200',  'mouse1_slice180',
            'mouse1_slice201', 'mouse1_slice212', 'mouse1_slice221', 'mouse1_slice232',
            'mouse1_slice241', 'mouse1_slice260', 'mouse1_slice271', 'mouse1_slice251',
            'mouse1_slice283', 'mouse1_slice291', 'mouse1_slice301', 'mouse1_slice313', 
            'mouse1_slice326', 'mouse2_slice139',
            'mouse2_slice1', 'mouse2_slice31', 'mouse2_slice20', 'mouse2_slice10',
            'mouse2_slice40', 'mouse2_slice50', 'mouse2_slice61', 'mouse2_slice70',
            'mouse2_slice79', 'mouse2_slice90', 'mouse2_slice99', 'mouse2_slice109', 
            'mouse2_slice119', 'mouse2_slice129', 'mouse2_slice151', 'mouse2_slice160']
    seed = 222
    runner = Runner(args, seed, exps)
    runner.make_folders()
    runner.run()
