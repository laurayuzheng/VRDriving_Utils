'''
Author: Niall Williams 
Description: File for computing trajectory curvature values. 

Computation and metrics based on SIGGRAPH course and this YouTube video: https://www.youtube.com/watch?v=8JCR6z3GLVI

'''

import os, sys, glob
from pickletools import read_unicodestring1
import numpy as np
import pandas as pd
import pathlib
import math

class Polyline():
    def __init__(self, data_file):
        self.verts = self.read_path_file(data_file)
        self.sum_metrics(data_file)
        self.sinuosity = self.compute_sinuosity()
        print("User: ", self.user_id)
        print("Scenario: ", self.scenario_num)
        print('Sinuosity:', self.sinuosity)
        self.curvature = self.compute_curvature()
        print()


    def read_path_file(self, data_file):
        verts = []
        self.path_length = 0
        df = pd.read_csv(data_file, sep=',', index_col=None, header=0)
        self.user_id = df.iloc[0]["user_id"]

        cols = [col for col in df]
        for index, row in df.iterrows():
            verts.append(np.array([float(row['transform_x']), float (row['transform_y'])]))
            if len(verts) > 1:
                d = np.linalg.norm(verts[-1] - verts[-2])
                self.path_length += d
        return verts
    
    def sum_metrics(self, data_file):
        steering_sum = 0
        throttle_sum = 0
        brake_sum = 0 

        # self.path_length = 0
        df = pd.read_csv(data_file, sep=',', index_col=None, header=0)
        self.user_id = df.iloc[0]["user_id"]
        self.scenario_num = df.iloc[0]["scenario_num"]

        cols = [col for col in df]
        for index, row in df.iterrows():
            steering_sum += abs(float(row['steering_input']))
            throttle_sum += float(row['throttle_input'])
            brake_sum += float(row['brake_input'])
        
        self.steering_sum = steering_sum
        self.throttle_sum = throttle_sum
        self.brake_sum = brake_sum

    def write_totals_to_csv(self, data_file, steer, brake, throttle):
        df = pd.read_csv(data_file, sep=',', index_col=None, header=0)

        # create columns if not exists
        steer_total_col = "steer_total"
        brake_total_col = "brake_total"
        throttle_total_col = "throttle_total"

        df[steer_total_col] = steer  
        df[brake_total_col] = brake
        df[throttle_total_col] = throttle 

        df.to_csv(data_file, encoding='utf-8', index=False)

    def write_to_personality_csv(self, data_file):
        df = pd.read_csv(data_file, sep=',', index_col=None, header=0)

        # create columns if not exists
        sinuosity_col = "sinuosity_%d" % (self.scenario_num)
        turning_angle_curvature_col = "turning_angle_curvature_%d" % (self.scenario_num)
        length_variation_curvature_col = "length_variation_curvature_%d" % (self.scenario_num)
        steiner_formula_curvature_col = "steiner_formula_curvature_%d" % (self.scenario_num)
        osculating_circle_curvature_col = "osculating_circle_curvature_%d" % (self.scenario_num)
        
        steering_sum_col = "steering_sum_%d" % (self.scenario_num)
        throttle_sum_col = "throttle_sum_%d" % (self.scenario_num)
        brake_sum_col = "brake_sum_%d" % (self.scenario_num)

        df[sinuosity_col] = df.get(sinuosity_col, 0)  
        df[turning_angle_curvature_col] = df.get(turning_angle_curvature_col, 0)  
        df[length_variation_curvature_col] = df.get(length_variation_curvature_col, 0) 
        df[steiner_formula_curvature_col] = df.get(steiner_formula_curvature_col, 0) 
        df[osculating_circle_curvature_col] = df.get(osculating_circle_curvature_col, 0) 
        df[steering_sum_col] = df.get(steering_sum_col, 0) 
        df[throttle_sum_col] = df.get(throttle_sum_col, 0) 
        df[brake_sum_col] = df.get(brake_sum_col, 0) 

        # update column values
        df.loc[self.user_id, sinuosity_col] = self.sinuosity
        df.loc[self.user_id, turning_angle_curvature_col] = self.turning_angle_curvature()
        df.loc[self.user_id, length_variation_curvature_col] = self.length_variation_curvature()
        df.loc[self.user_id, steiner_formula_curvature_col] = self.steiner_formula_curvature()
        df.loc[self.user_id, osculating_circle_curvature_col] = self.osculating_circle_curvature()
        df.loc[self.user_id, steering_sum_col] = self.steering_sum
        df.loc[self.user_id, throttle_sum_col] = self.throttle_sum
        df.loc[self.user_id, brake_sum_col] = self.brake_sum

        df.to_csv(data_file, encoding='utf-8', index=False)

    def compute_sinuosity(self):
        start_to_end = np.linalg.norm(self.verts[0] - self.verts[-1])
        return start_to_end / self.path_length

    def compute_curvature(self):
        # Grinspun, Eitan, and Adrian Secord. "Introduction to discrete differential geometry: the geometry of plane curves." ACM SIGGRAPH ASIA 2008 courses. 2008. 1-4.
        #  Lecture 1: Overview (Discrete Differential Geometry): https://www.youtube.com/watch?v=8JCR6z3GLVI
        print('Turning angle curvature:', self.turning_angle_curvature())
        print('Length variation curvature:', self.length_variation_curvature())
        print('Steiner formula curvature:', self.steiner_formula_curvature())
        print('Osculating circle curvature:', self.osculating_circle_curvature())

    def turning_angle_curvature(self):
        all_thetas = []
        for i in range(1, len(self.verts) - 1):
            p1 = self.verts[i-1]
            p2 = self.verts[i]
            p3 = self.verts[i+1]
            all_thetas.append(self.angle_between_vectors(p1, p2, p3))
        return sum(all_thetas) / len(all_thetas)

    def length_variation_curvature(self):
        all_gradients = []
        for i in range(1, len(self.verts) - 1):
            p1 = self.verts[i-1]
            p2 = self.verts[i]
            p3 = self.verts[i+1]
            exterior_angle = self.angle_between_vectors(p1, p2, p3)
            all_gradients.append(2 * math.sin(exterior_angle * 0.5))
        return sum(all_gradients) / len(all_gradients)

    def steiner_formula_curvature(self):
        all_gradients = []
        for i in range(1, len(self.verts) - 1):
            p1 = self.verts[i-1]
            p2 = self.verts[i]
            p3 = self.verts[i+1]
            exterior_angle = self.angle_between_vectors(p1, p2, p3)
            all_gradients.append(2 * math.tan(exterior_angle * 0.5))
        return sum(all_gradients) / len(all_gradients)

    def osculating_circle_curvature(self):
        all_radii = []
        for i in range(1, len(self.verts) - 1):
            p1 = self.verts[i-1]
            p2 = self.verts[i]
            p3 = self.verts[i+1]
            exterior_angle = self.angle_between_vectors(p1, p2, p3)
            w_i = np.linalg.norm(p1 - p3)
            if w_i < 0.0001:
                continue
            all_radii.append(2 * math.sin(exterior_angle) / w_i)
        return sum(all_radii) / len(all_radii)

    def angle_between_vectors(self, p1, p2, p3):
        v1_length = np.linalg.norm(p2 - p1)
        v2_length = np.linalg.norm(p3 - p2)
        if v1_length < 0.0001 or v2_length < 0.0001:
            return 0

        v1 = (p2 - p1) / v1_length
        v2 = (p3 - p2) / v2_length
        return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

if __name__ == '__main__':
    DATADIR = "./csv_output"

    sim_file_paths = glob.glob(os.path.join(DATADIR, "simdata", "*.csv"))
    questionnaire_csv_file_path = os.path.join(DATADIR, "questionnaire_processed.csv")

    user_steer = np.zeros((19))
    user_throttle = np.zeros((19))
    user_brake = np.zeros((19))

    for data_file in sim_file_paths:
        path = Polyline(data_file)

        if path.scenario_num not in [0,1,2,3] or "CARLAROUTE" in data_file:
            continue 
        
        user_steer[path.user_id] += path.steering_sum
        user_throttle[path.user_id] += path.throttle_sum
        user_brake[path.user_id] += path.brake_sum

        path.write_to_personality_csv(questionnaire_csv_file_path)
    
    path.write_totals_to_csv(questionnaire_csv_file_path, user_steer, user_brake, user_throttle)
    
