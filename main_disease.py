"""
Created on April 24, 2024
@author: Zhuo Li

The main script of trainning and evaluating DQN for AD Seeking
Without state normalization
Dataset: Nature Neuralcomputing Disease
"""
import time
import random
import os
import numpy as np
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

from env.env_dis_tau import SpatOmics_dis
# from env.env_dis_both import SpatOmics_dis
from src.dqn import DQN
from src.args import get_dqn_args


class Runner:
    def __init__(self, args, seed, categ, pos, exps):
        self.args = args
        self.seed = seed
        self.exps = exps
        self.categ = categ
        self.pos = pos
        self.num = len(categ)
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.grid_x, self.grid_y = self.args.grid_x, self.args.grid_y
        # observation and action spaces
        n_obs = self.grid_x*self.grid_y*self.args.cell_num + 2 + 17
        self.observation_space = spaces.Box(low = -np.array(np.ones(n_obs)), high = np.array(np.ones(n_obs)), dtype=np.float64)
        self.action_space = spaces.Discrete(24)
        self.args.obs_dim = self.observation_space.shape[0]  # obs dimensions
        self.args.action_dim = self.action_space.n  # actions dimensions
        # Create an agent
        self.agent = DQN(self.args.obs_dim, self.args.action_dim, args)

        self.writer = SummaryWriter(log_dir='runs/{}/seed_{}'.format(self.args.agent_name, self.seed)) # Create a tensorboard
        self.train_rewards = [] # Record the rewards during the training
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.best_eval_rew = - float("Inf") # Initialize the best reward

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
            id_exp = random.randint(0, self.num-1)
            env = SpatOmics_dis(self.args, self.categ[id_exp], self.pos[id_exp], self.exps[id_exp])  # self.exps[id_exp] 
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
        id_exp = random.randint(0, self.num-1)
        env_evaluate = SpatOmics_dis(self.args, self.categ[id_exp], self.pos[id_exp], self.exps[id_exp])  # self.exps[id_exp],
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            state = env_evaluate.reset()
            episode_reward = 0
            obses = []
            AD_countses= []
            for eval_time in range(self.args.episode_size):
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
    exps = ['8months-disease-replicate_1', '8months-disease-replicate_2', '13months-disease-replicate_1']  # Abeta
    categ, pos = [], []
    for exp in exps:
    # exp= exps
        file_path = os.path.join('dataset_disease', 'spatial_{}.csv'.format(exp))
        data_pos, data_categ = [], []
        with open(file_path, newline='') as csvfile:
            # 创建一个csv读取器
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                temp = row[3:5]
                temp_num = [int(x) for x in temp]
                data_pos.append(temp_num)
                data_categ.append(row[6:7])
        pos.append(np.array(data_pos))
        categ.append(np.array(data_categ))

    seed = 594   
    runner = Runner(args, seed, categ, pos, exps)
    runner.make_folders()
    runner.run()
