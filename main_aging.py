"""
Created on Mar 18, 2024
@author: Weiran Wu

The main script of trainning and evaluating the DQN algorithm for AD Seeking
"""

import time
import random
import os
import numpy as np
import torch
import os
import csv
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

from env.env_aging import SpatOmics_dis
from src.dqn import DQN
from src.args import get_dqn_args

class Runner:
    def __init__(self, args, seed, categ, pos, age, exps):
        self.args = args
        self.seed = seed
        self.categ = categ
        self.pos = pos
        self.age = age
        self.num = len(categ)
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


        self.writer = SummaryWriter(log_dir='runs/{}/seed_{}'.format(self.args.agent_name, self.seed)) # Create a tensorboard
        self.train_rewards = [] # Record the rewards during the training
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.best_eval_rew = - float("Inf") # Initialize the best reward

    def make_folders(self, ):
        data_folder = 'data'
        model_folder = 'model'
        data_seed_folder = os.path.join(data_folder, f'{self.args.agent_name}', f'seed_{self.seed}')
        self.model_seed_folder = os.path.join(model_folder, f'{self.args.agent_name}', f'seed_{self.seed}')
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
            env = SpatOmics_dis(self.args, self.age, self.categ[id_exp], self.pos[id_exp], self.exps[id_exp])
            state = env.reset()
            epi_num += 1
            epi_steps = 0
            episode_reward = 0
            time_episode_start = time.time()
            for _ in range(self.args.episode_size): 
                action = self.agent.select_action(state)
                next_state, reward, done, _ = env.step(action)  # Step
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
            print("Training: Episode:{}, Total steps: {}, epi steps: {}, epi rewards: {}, epi time: {}".
                  format(epi_num, self.total_steps, epi_steps, round(episode_reward,2), round((time_episode_end - time_episode_start),2)))

            self.writer.add_scalar('train_rewards', episode_reward, global_step=self.total_steps)
            self.train_rewards.append(episode_reward)
            np.save('./{}/train_episode_reward.npy'.format(self.data_train_folder), self.train_rewards)


    def evaluate_policy(self, ):
        # Create env
        id_exp = random.randint(0, self.num-1)
        env_evaluate = SpatOmics_dis(self.args, self.age, self.categ[id_exp], self.pos[id_exp], self.exps[id_exp])
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            state = env_evaluate.reset()
            episode_reward = 0
            obses = []
            AD_countses= []
            for eval_time in range(self.args.episode_size):
                action = self.agent.select_action(state, isEval=True)
                next_state, reward, done, _ = env_evaluate.step(action)  # Step
                state = next_state
                episode_reward += reward
                AD_countses.append(episode_reward)
                obs_x = state[0]
                obs_y = state[1]
                obses.append([obs_x, obs_y])
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



def load_data(exps):
    categ, pos = [], []
    for exp in exps:
        file_path = os.path.join('dataset_Aging', '{}'.format(age), '{}_cell_type_annot.csv'.format(exp))
        data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                data.append(row)
        categ.append(np.array(data))


        file_path = os.path.join('dataset_Aging', '{}'.format(age), '{}_pos.csv'.format(exp))
        data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                temp = [float(cell) for cell in row]
                data.append(temp)
        pos.append(np.array(data)) # np.array(data.values.tolist())

    return categ, pos


if __name__ == '__main__':
    args = get_dqn_args()

    args.rs = 200
    args.cell_num = 13

    # print("Start running {} for Sampling in Spatial Omics".format(args.agent_name))
    seed = 333
    ages = ('4wk', '24wk', '90wk', 'all')
    for age in ages:
        seed = seed + 1
        if age == '4wk':
            exps = ['MsBrainAgingSpatialDonor_1_0', 'MsBrainAgingSpatialDonor_8_0', 'MsBrainAgingSpatialDonor_4_1',
                    'MsBrainAgingSpatialDonor_4_2', 'MsBrainAgingSpatialDonor_7_0', 'MsBrainAgingSpatialDonor_7_2',
                'MsBrainAgingSpatialDonor_4_0', 'MsBrainAgingSpatialDonor_8_1']   ## only for S1, S3
            # 'MsBrainAgingSpatialDonor_7_1', 'MsBrainAgingSpatialDonor_8_2'   ## testing set
            categ, pos = load_data(exps)
            runner = Runner(args, seed, categ, pos, age, exps)
            runner.make_folders()
            runner.run()
        elif age == '24wk':
            exps = ['MsBrainAgingSpatialDonor_10_0', 'MsBrainAgingSpatialDonor_10_2',
            'MsBrainAgingSpatialDonor_11_0', 'MsBrainAgingSpatialDonor_12_0', 'MsBrainAgingSpatialDonor_12_1', 
            'MsBrainAgingSpatialDonor_11_2']   ## only for S1, S3
            # 'MsBrainAgingSpatialDonor_11_1', 'MsBrainAgingSpatialDonor_10_1'   ## testing set
            categ, pos = load_data(exps)
            runner = Runner(args, seed, categ, pos, age, exps)
            runner.make_folders()
            runner.run()
        elif age == '90wk':
            exps = ['MsBrainAgingSpatialDonor_2_0', 'MsBrainAgingSpatialDonor_2_1',
            'MsBrainAgingSpatialDonor_3_0', 'MsBrainAgingSpatialDonor_3_1', 'MsBrainAgingSpatialDonor_9_0',
            'MsBrainAgingSpatialDonor_5_0', 'MsBrainAgingSpatialDonor_5_2', 'MsBrainAgingSpatialDonor_9_1',
            'MsBrainAgingSpatialDonor_6_0', 'MsBrainAgingSpatialDonor_5_1', ## only for S1, S3
            'MsBrainAgingSpatialDonor_9_1', 'MsBrainAgingSpatialDonor_6_2']
            # 'MsBrainAgingSpatialDonor_6_1', 'MsBrainAgingSpatialDonor_9_2'   ## testing set
            categ, pos = load_data(exps)
            runner = Runner(args, seed, categ, pos, age, exps)
            runner.make_folders()
            runner.run()
        else:
            exps = ['MsBrainAgingSpatialDonor_1_0', 'MsBrainAgingSpatialDonor_8_0', 'MsBrainAgingSpatialDonor_4_1',
                    'MsBrainAgingSpatialDonor_4_2', 'MsBrainAgingSpatialDonor_7_0', 'MsBrainAgingSpatialDonor_7_2',
                    'MsBrainAgingSpatialDonor_4_0', 'MsBrainAgingSpatialDonor_8_1', 'MsBrainAgingSpatialDonor_10_0', 
                    'MsBrainAgingSpatialDonor_10_2',
                    'MsBrainAgingSpatialDonor_11_0', 'MsBrainAgingSpatialDonor_12_0', 'MsBrainAgingSpatialDonor_12_1', 
                    'MsBrainAgingSpatialDonor_11_2','MsBrainAgingSpatialDonor_2_0', 'MsBrainAgingSpatialDonor_2_1',
                    'MsBrainAgingSpatialDonor_3_0', 'MsBrainAgingSpatialDonor_3_1', 'MsBrainAgingSpatialDonor_9_0',
                    'MsBrainAgingSpatialDonor_5_0', 'MsBrainAgingSpatialDonor_5_2', 'MsBrainAgingSpatialDonor_9_1',
                    'MsBrainAgingSpatialDonor_6_0', 'MsBrainAgingSpatialDonor_5_1', ## only for S1, S3
                    'MsBrainAgingSpatialDonor_6_2']
            categ, pos = load_data(exps)
            runner = Runner(args, seed, categ, pos, age, exps)
            runner.make_folders()
            runner.run()
