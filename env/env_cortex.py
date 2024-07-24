# -*- coding: utf-8 -*-
"""
Created on March 13 2024
@author: Weiran Wu

Samping in Spatio Omics: 
without state normalization
24 dimensional discrete action, 
penalty for overlapped area
"""

import numpy as np
from gym import spaces
import csv
import random
import os


class SpatOmics_dis():
    def __init__(self, args, exp):
        self.grid_x, self.grid_y = args.grid_x, args.grid_y
        self.cell_num = args.cell_num
        ## load raw data from csv
        file_path = os.path.join('dataset_cortex', '{}_categ.csv'.format(exp))
        data = []

        with open(file_path, newline='') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                data.append(row)
        self.categ = data # np.array(data.values.tolist())
                
        file_path = os.path.join('dataset_cortex', '{}_pos.csv'.format(exp))
        data = []

        with open(file_path, newline='') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                temp = [float(cell) for cell in row]
                data.append(temp)
        self.pos = np.array(data) # np.array(data.values.tolist())

        # set range
        self.x_max = np.amax(self.pos[:,0])
        self.x_min = np.amin(self.pos[:,0])
        self.y_max = np.amax(self.pos[:,1])
        self.y_min = np.amin(self.pos[:,1])
        # set grid size
        self.grid_width = 50
        self.grid_height = 50

        # count grids
        self.map_x_range = int(np.ceil((self.x_max - self.x_min) / self.grid_width))
        self.map_y_range = int(np.ceil((self.y_max - self.y_min) / self.grid_height))

        ## generate AD
        mark = np.array(self.categ) == 'L45_IT'  # find L45_IT mark
        mark_indices = np.where(np.array(self.categ) == 'L45_IT')[0] # indices of mark
        self.mark_indices = mark_indices

        self.AD = []
        for i in range(mark.shape[0]):
            if mark[i]:
                if np.random.rand() < 0.3:
                    new_point = self.pos[i] + np.random.rand(2) - 0.5
                    radius = np.random.uniform(5, 10)
                    new_point = np.append(new_point, radius)
                    self.AD.append(new_point)

        if len(self.AD) < 1:
            i=(mark.shape[0])
            if mark[i]:
                new_point = self.pos[i] + np.random.rand(2) - 0.5
                radius = np.random.uniform(5, 10)
                new_point = np.append(new_point, radius)
                self.AD.append(new_point)

        self.AD = np.array(self.AD)
        self.mark_pos = self.pos[self.mark_indices, :]
        self.AD_pos = self.AD[:, :2]

        # other cells
        self.cells_pos = []
        for cell in ['L45_IT', 'L23_IT', 'L56_NP', 'L5_IT', 'L5_PT', 'L6_CT', 'L6_IT', 'Astrocytes',
                     'Endothelial', 'L6_IT_Car3', 'L6b', 'Oligodendrocytes', 'OPC', 'Microglia', 'Lamp5', 
                     'VLMC', 'Sst', 'SMC', 'Pvalb', 'Pericytes', 'PVM']:
            celli_indices = np.where(np.array(self.categ) == cell)[0]  # indices of mark
            celli_pos = self.pos[celli_indices, :]
            self.cells_pos.append(celli_pos)
        # initialize model parameters
        # self.initialize_params()
        self.rs = args.rs
      
    # def initialize_params(self):
    #     self.rs = 200
        # observation and action spaces
        n_obs = self.grid_x*self.grid_y*self.cell_num + 2 + 17
        self.observation_space = spaces.Box(low = -np.array(np.ones(n_obs)), high = np.array(np.ones(n_obs)), dtype=np.float32)
        self.action_space = spaces.Discrete(8+16)
        
    def step(self, action, rand=False, isEval=False):
        self.count += 1
        action = action + 1 
        x, y = self.pos_sampling

        if rand:
            next_x, next_y = (round(random.uniform(self.x_min, self.x_max), 1), round(random.uniform(self.y_min, self.y_max), 1))
        else:
            if action < 9:
                if action in [1, 4, 6]:
                    next_x = x - self.rs
                elif action in [2, 7]:
                    next_x = x
                else:
                    next_x = x + self.rs
                if action in [1, 2, 3]:
                    next_y = y + self.rs
                elif action in [4, 5]:
                    next_y = y
                else:
                    next_y = y - self.rs
            else:
                if 9 <= action <= 13:
                    next_x = x + (action-11) *self.rs
                    next_y = y + 2 *self.rs
                elif action in [14, 15, 16]:
                    next_x = x + 2 *self.rs
                    next_y = y + (15-action) *self.rs
                elif action in [17, 18, 19]:
                    next_x = x - 2 *self.rs
                    next_y = y + (18-action) *self.rs
                else:
                    next_x = x + (action-22) *self.rs
                    next_y = y - 2 *self.rs

        # projection to the boundary
        next_x = self.x_min + self.rs/2 if next_x < self.x_min + self.rs/2 else next_x
        next_x = self.x_max - self.rs/2 if next_x > self.x_max - self.rs/2 else next_x
        next_y = self.y_min + self.rs/2 if next_y < self.y_min + self.rs/2 else next_y
        next_y = self.y_max - self.rs/2 if next_y > self.y_max - self.rs/2 else next_y
        self.pos_sampling = [next_x, next_y]

        ## get reward
        # reward for overlapped area
        self.update_map()
        r_overlap = self.get_overlap_area()/1000
        # reward for AD
        cell_counts, AD_counts = self.measure()
        mk = np.array(cell_counts).flatten() 
        r_AD = np.sum(AD_counts)

        if isEval:
            reward = 5 * r_AD
        else:
            reward = 1*r_overlap + 5*r_AD
            if r_AD > 2:
                 self.success += 1
            if self.success > 9:
                self.done = True
                reward += 100

        self.samp_corner_store.append((next_x-self.rs/2, next_y-self.rs/2, next_x+self.rs/2, next_y+self.rs/2, 0))

        ## get state
        row_sampling = int((next_x - self.x_min) / self.grid_width) - 1
        col_sampling = int((next_y - self.y_min) / self.grid_height) - 1
        abs_map = [1] + self.grid_far(row_sampling, col_sampling) + self.grid_around(row_sampling, col_sampling)
        state = np.concatenate((np.array([next_x, next_y]), np.array(abs_map), mk))

        return state, reward, self.done, r_AD

    def update_map(self):

        x_min_index = max(0, int((self.pos_sampling[0] - self.rs - self.x_min) / self.grid_width))
        x_max_index = min(self.map_x_range-1, int((self.pos_sampling[0] + self.rs - self.x_min) / self.grid_width))
        y_min_index = max(0, int((self.pos_sampling[1] - self.rs - self.y_min) / self.grid_height))
        y_max_index = min(self.map_y_range-1, int((self.pos_sampling[1] + self.rs - self.y_min) / self.grid_height))


        for i in range(x_min_index, x_max_index+1):
            for j in range(y_min_index, y_max_index+1):
                self.map[i][j] += 1   

    def measure(self):
        x_s = self.pos_sampling[0]
        y_s = self.pos_sampling[1]

        cell_counts = []
        for i in range(len(self.cells_pos)):
            celli_counts = self.count_points_in_grid(x_s, y_s, self.grid_x, self.grid_y, self.cells_pos[i])
            cell_counts.append(celli_counts)
        AD_counts = self.count_points_in_grid(x_s, y_s, self.grid_x, self.grid_y, self.AD_pos)

        return cell_counts, AD_counts

    def count_points_in_grid(self, x_s, y_s, grid_x, grid_y, points):
        grid_counts = np.zeros((grid_x, grid_y))
        rs = self.rs
        grid_size =  rs/grid_x 
        xs_min = max(x_s - rs/2, self.x_min)
        ys_min = max(y_s - rs/2, self.y_min)
        xs_max = min(x_s + rs/2, self.x_max)
        ys_max = min(y_s + rs/2, self.y_max)
        for p in points:
            if xs_min <= p[0] <= xs_max and ys_min <= p[1] <= ys_max:
                grid_i = min(int((p[0] - xs_min) // grid_size), grid_x-1)
                grid_j = min(int((p[1] - ys_min) // grid_size), grid_y-1)
                grid_counts[grid_j, grid_i] += 1

        return grid_counts

    def get_overlap_area(self):
        reward = 0
        x_center, y_center = self.pos_sampling[0], self.pos_sampling[1]
        for i, (x1, y1, x2, y2, n) in enumerate(self.samp_corner_store):
            overlap_x1 = max(x_center - self.rs / 2, x1)
            overlap_x2 = min(x_center + self.rs / 2, x2)
            overlap_y1 = max(y_center - self.rs / 2, y1)
            overlap_y2 = min(y_center + self.rs / 2, y2)
            overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
            if overlap_area > 0:
                self.samp_corner_store[i] = (x1, y1, x2, y2, n+1)
            reward += overlap_area * (n+1)
        return -reward

 
    def grid_around(self, row, col):
        around_grids = [0 for _ in range(8)]
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row+dr, col+dc
            if 0 <= new_row < self.map_x_range and 0 <= new_col < self.map_y_range:
                around_grids[i] = self.map[new_row][new_col]
            else:
                around_grids[i] = 0
        
        return around_grids
    
    def grid_far(self, row, col):

        average_grids = [0 for _ in range(8)]
        num_grids = [0 for _ in range(8)]
        for i in range(self.map_x_range):
            for j in range(self.map_y_range):
                if i < row-1 and j < col-1:
                    average_grids[0] += self.map[i][j]
                    num_grids[0] += 1
                elif i in [row-1,row,row+1] and j < col-1:
                    average_grids[1] += self.map[i][j]
                    num_grids[1] += 1
                elif i > row+1 and j < col-1:
                    average_grids[2] += self.map[i][j]
                    num_grids[2] += 1
                elif i < row-1 and j in [col-1,col,col+1]:
                    average_grids[3] += self.map[i][j]
                    num_grids[3] += 1
                elif i > row+1 and j in [col-1,col,col+1]:
                    average_grids[4] += self.map[i][j]
                    num_grids[4] += 1
                elif i < row-1 and j > col+1:
                    average_grids[5] += self.map[i][j]
                    num_grids[5] += 1
                elif i in [row-1,row,row+1] and j > col+1:
                    average_grids[6] += self.map[i][j]
                    num_grids[6] += 1
                elif i > row+1 and j > col+1:
                    average_grids[7] += self.map[i][j]
                    num_grids[7] += 1
        for k in range(8):
            average_grids[k] = average_grids[k]/num_grids[k] if num_grids[k]>0 else 0 
        return average_grids


    def reset(self):
        self.count = 0
        self.success = 0
        self.done = False
        self.samp_corner_store = []
        self.map = [[0 for _ in range(self.map_y_range)] for _ in range(self.map_x_range)]
        ## reset the initial sampling position
        x, y = round(random.uniform(self.x_min, self.x_max), 1), round(random.uniform(self.y_min, self.y_max), 1) # random
        # x, y = self.x_min + self.rs/2, self.y_min + self.rs/2  # warm start
        self.pos_sampling = [x, y]
        cell_counts, AD_counts = self.measure()

        # if np.sum(AD_counts) == 0:
        #     self.pos_sampling = self.warm_start_b(x, y)
        #     cell_counts, AD_counts = self.measure()
        # x, y = self.pos_sampling[0], self.pos_sampling[1]

        self.samp_corner_store.append((x-self.rs/2, y-self.rs/2, x+self.rs/2, y+self.rs/2, 0))
        self.update_map()
        mk = np.array(cell_counts).flatten()   
        row_sampling = int((self.pos_sampling[0] - self.x_min) / self.grid_width) - 1
        col_sampling = int((self.pos_sampling[1] - self.y_min) / self.grid_height)- 1
        abs_map = [1] + self.grid_far(row_sampling, col_sampling) + self.grid_around(row_sampling, col_sampling)
        state = np.concatenate((np.array([x, y]), np.array(abs_map), mk))


        return state
    
    def warm_start_b(self, x, y):
        sampling_ended = False
        while x < self.x_max - self.rs/2:
            x = x + self.rs
            y = self.y_min + self.rs/2
            while y < self.y_max - self.rs/2:
                y = y + self.rs
                self.pos_sampling = [x, y]
                _, AD_counts = self.measure()
                if np.sum(AD_counts) > 0:
                    sampling_ended = True
                    break
            if sampling_ended:
                break
        
        return self.pos_sampling