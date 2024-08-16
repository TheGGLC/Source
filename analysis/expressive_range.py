# from https://github.com/riffsircar/blend-elites/blob/master/metrics.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import math
import os
import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def density(segment):
    total = 0
    for l in segment:
        total += len(l)-l.count('-')
    return total

def get_level_and_solution(path):
    level = []
    level_str = ""
    difficalty = None
    with open(path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            if not line.startswith("META"):
                ncoded_line = [x for x in line]
                level.append(ncoded_line)
                level_str += line
            else:
                try:

                    json_str = line.split("META", 1)[1].strip()
                    meta_obj = json.loads(json_str)
                    if meta_obj.get("shape") == "path":
                        difficalty =  meta_obj.get("data")
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")
    return level, level_str, difficalty

def calculate_path_length(segments):
    total_length = 0.0
    
    for segment in segments:
        x1, y1, x2, y2 = segment
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_length += distance
    
    return total_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='mario')
    opt = parser.parse_args()

    game = opt.game
    parent_directories = [f'./TheGGLC/{game}/solvable/texts']

    report = pd.DataFrame(columns=[
                            'density',
                            'nonlinearity'])
    report.reset_index(drop=True, inplace=True)


    for parent_directory in parent_directories:
        for _, dirs, _ in os.walk(parent_directory):
            for dir in dirs:
                dir_path = os.path.join(parent_directory, dir)
                for _, _, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(dir_path, file)

                        level_array, level, data_value = get_level_and_solution(file_path)
                        density_level = density(level)
                        if data_value is not None:
                            d = {   
                                    'density': density_level,
                                    'difficalty': calculate_path_length(data_value),
                                    }
                            # print(d)
                            new_row = pd.DataFrame([d])
                            report = pd.concat([report, new_row], ignore_index=True)

    report.to_csv(f'./out/{game}_expressive.csv', index=False)

    scaler = MinMaxScaler()
    report[['density', 'difficalty']] = scaler.fit_transform(report[['density', 'difficalty']])

    # Plot the normalized data
    plt.figure(figsize=(8, 6))
    plt.hexbin(report['density'], report['difficalty'], gridsize=50, cmap='viridis')  # Create a hexbin plot (heatmap)
    plt.colorbar(label='count in bin')  # Add a colorbar
    plt.xlabel('density')
    plt.ylabel('difficalty')
    plt.title('Heatmap of Normalized density and difficalty')
    plt.savefig(f'./out/density_difficalty_{game}_normalized.png')