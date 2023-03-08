import numpy as np

import random

from run_stats import * 
from config import * 

import pandas as pd
import matplotlib.pyplot as plt
from math import pi

VARS = [
    '         Reckless', 
    'Anxiety',
    'Risk-taking',
    'Anger',
    'High \n Velocity   ',
    'Distress   \n Reduction',
    'Patience',
    '\n  Carefullness',
]

def plot_single_personality(data):
    
    # Categories were originally from df.columns
    # categories = list(data.columns)
    categories = VARS
    N = len(categories)
   
   # Repeat the first value to close the circular graph:
    values = (data.values.flatten() * 100).tolist()
    values += values[:1]
    
    # angle for each value in the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,30)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Personality A")
 
    # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.1))

    # Show the graph
    plt.show()


    return None


def plot_two_personalities(p1, p2):

    categories = VARS
    N = len(categories)

    # angle for each value in the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,30)

    # plot first personality
    values = (p1.values.flatten() * 100).tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Personality A")
    ax.fill(angles, values, 'blue', alpha=0.1)

    # plot second personality
    values = (p2.values.flatten() * 100).tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1 ,linestyle='solid', label="Personality B")
    ax.fill(angles, values, 'red', alpha=0.1)

    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.05, 0.1))

    # Show the graph
    plt.show()

    
    
    return None

def plot_multiple_personalities(data):
    categories = VARS
    N = len(categories)

    # angle for each value in the radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
    plt.ylim(0,25)

    # find number of personalities
    num_personalities = len(data)
    print(num_personalities)

    abet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    # plot each personality
    for i in range(num_personalities):

        values = (data.iloc[i].values.flatten() * 100).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=f"Personality {abet[i]}")
        ax.fill(angles, values, alpha=0.1)

    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # Show the graph
    plt.show()
    
    return None


if __name__ == "__main__":
    stats = StatsManager(DATADIR, EXCLUSIONS)

    df = stats.get_personality_data()

    print(df.tail(1))
    # baselines for scenarios
    # index 14
    df.loc[len(df.index)] = [0.1163, 0.1105, 0.0924, 0.1069, 0.1410, 0.1310, 0.1907, 0.2149]
    # index 15
    df.loc[len(df.index)] = [0.1576, 0.1185, 0.0915, 0.2230, 0.1370, 0.0366, 0.1656, 0.1432]
    # index 16
    df.loc[len(df.index)] = [0.0826, 0.0816, 0.1615, 0.1047, 0.1322, 0.1292, 0.1499, 0.2323]
    # index 17
    df.loc[len(df.index)] = [0.0977, 0.0753, 0.0928, 0.0923, 0.1423, 0.1395, 0.1787, 0.1658]

    # plot_single_personality(df.iloc[17])
    # plot_two_personalities(df.iloc[1], df.iloc[14])
    plot_multiple_personalities(df.tail(4))
