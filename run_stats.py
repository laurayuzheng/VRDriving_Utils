''' 
Run statistics on collected CSV data. 
Created by Laura Zheng 
Feb 16, 2023
'''

import sys 
import os, glob

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from sklearn.manifold import TSNE

MDSI_COLUMNS = [
    'score_reckless', 
    'score_anxious',
    'score_risky',
    'score_angry',
    'score_high_velocity',
    'score_distress_reduction',
    'score_patient',
    'score_careful'
]

DISSOCIATIVE_COLUMNS = [
    "misjudge the speed of an oncoming vehicle when passing",
    "intend to switch on the windscreen wipers, but switch on the lights instead",
    "forget that my lights are on full beam until flashed by another motorist",
    "nearly hit something due to misjudging my gap in a parking lot", 
    "plan my route badly, so that I hit traffic that I could have avoided", 
    "attempt to drive away from traffic lights in third gear (or on the neutral mode in automatic cars)",
    "lost in thoughts or distracted, I fail to notice someone at the pedestrian crossings",
    "I daydream to pass the time while driving"
]

ANXIOUS_COLUMNS = [
    "feel nervous while driving", 
    "feel distressed while driving",
    "driving makes me feel frustrated",
    "it worries me when driving in bad weather",
    "on a clear freeway, I usually drive at or a little below the speed limit", 
    "feel I have control over driving", # -1
    "feel comfortable while driving" # -1
]

RISKY_COLUMNS = [
    "enjoy the excitement of dangerous driving", 
    "enjoy the sensation of driving on the limit", 
    "like to take risks while driving", 
    "like the thrill of flirting with death or disaster", 
    "fix my hair/ makeup while driving"
]

ANGRY_COLUMNS = [
    "swear at other drivers", 
    'blow my horn or “flash” the car in front as a way of expressing frustrations',
    "when someone does something on the road that annoys me, I flash them with the high beam", 
    "honk my horn at others", 
    "when someone tries to skirt in front of me on the road, I drive in an assertive way in order to prevent it"
]

HIGH_VELOCITY_COLUMNS = [
    "in a traffic jam, I think about ways to get through the traffic faster", 
    "when in a traffic jam and the lane next to me starts to move, I try to move into that lane as soon as possible", 
    "when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves", 
    "purposely tailgate other drivers", 
    "get impatient during rush hours", 
    "drive through traffic lights that have just turned red"
]

DISTRESS_REDUCTION_COLUMNS = [
    "use muscle relaxation techniques while driving", 
    "while driving, I try to relax myself", 
    "do relaxing activities while driving", 
    "meditate while driving"
]

PATIENT_COLUMNS = [
    "at an intersection where I have to give right-of-way to oncoming traffic, I wait patiently for cross-traffic to pass",
    'base my behavior on the motto “better safe than sorry”',
    "when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves", # -1
    "plan long journeys in advance" # -1
]

CAREFUL_COLUMNS = [
    "drive cautiously",
    "always ready to react to unexpected maneuvers by other drivers",
    "distracted or preoccupied, and suddenly realize the vehicle ahead has slowed down, and have to slam on the breaks to avoid a collision",
    "get a thrill out of breaking the law"
]

# Rotation matrix in counter clockwise direction 
def rotation_matrix(theta):
    theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]]
                    )

class StatsManager: 
    '''
    Stats class for data processing on VR driving study data. 
    '''

    def __init__(self, datadir, exclusions=[]):
        self.datadir = datadir 
        self.exclusions = exclusions 

        self.questionnaire_df = None 
        self.sim_dfs = None 

        self._get_data()
        self._vectorize_personalities()

    def _get_data(self):
        '''
        Gets data from self.datadir and updates self.questionnaire_df and self.sim_df.
        '''

        sim_data_csv_list = glob.glob(os.path.join(self.datadir, "sim_data/*/*.csv"))

        self.sim_dfs = [] 

        for filename in sim_data_csv_list:

            exp_id = "_".join(filename.split("/")[-2].split("_")[:3])
            user_id = int(exp_id.split("_")[0])
            scenario_num = int(exp_id.split("_")[2])

            # exclude user from data
            if user_id in self.exclusions:
                continue 

            df = pd.read_csv(filename, index_col=None, header=0)
            df.columns = df.columns.str.strip()

            df["exp_id"] = exp_id 
            df["user_id"] = user_id
            df["scenario_num"] = scenario_num

            # process data to numeric types 
            df["transform"] = df["transform"].apply(StatsManager._convert_transform_str_to_int)

            # Drop first few rows; sensor records info before initializing scenario
            df = df.tail(-2)
            df = df.reset_index(drop=True)

            first_x = df["transform"][0][0][0]
            first_y = df["transform"][0][0][1]
            first_z = df["transform"][0][0][2]

            # first_rot_x = df["transform"][0][1][0]
            # first_rot_y = df["transform"][0][1][1]
            first_rot_z = df["transform"][0][1][2]

            df["transform_x"] = df["transform"].apply(lambda x: x[0][0] - first_x)
            df["transform_y"] = df["transform"].apply(lambda x: x[0][1] - first_y)
            df["transform_z"] = df["transform"].apply(lambda x: x[0][2] - first_z)

            df["rotation_x"] = df["transform"].apply(lambda x: x[1][0])
            df["rotation_y"] = df["transform"].apply(lambda x: x[1][1])
            df["rotation_z"] = df["transform"].apply(lambda x: x[1][2])

            # Rotate trajectories so always driving forward 
            rot_matrix = rotation_matrix(180 - first_rot_z)

            for index, row in df.iterrows():

                xy_pos_vec = np.array([row["transform_x"], row["transform_y"]])
                xy_rot_vec = np.array([row["rotation_x"], row["rotation_y"]])

                xy_pos_vec = xy_pos_vec @ rot_matrix.T 
                xy_rot_vec = xy_rot_vec @ rot_matrix.T 

                df.loc[index,"normalized_pos_x"] = xy_pos_vec[0]
                df.loc[index,"normalized_pos_y"] = xy_pos_vec[1]
                df.loc[index,"normalized_rot_x"] = xy_rot_vec[0]
                df.loc[index,"normalized_rot_y"] = xy_rot_vec[1]

            self.sim_dfs.append(df)
            print(exp_id)

        self.questionnaire_df = pd.read_csv(os.path.join(self.datadir, "questionnaire.csv"), index_col=None, header=0)
        self.questionnaire_df.columns = self.questionnaire_df.columns.str.replace(".1", "").str.strip()
        self.questionnaire_df.rename({'What is your gender?': 'gender'}, axis=1, inplace=True)

        # Invert the scores for some questions for analysis (based on original MDSI paper)
        self.questionnaire_df["feel I have control over driving"] = 6 - self.questionnaire_df["feel I have control over driving"]
        self.questionnaire_df["feel comfortable while driving"] = 6 - self.questionnaire_df["feel comfortable while driving"]
        self.questionnaire_df["when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves"] = 6 - self.questionnaire_df["when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves"]
        self.questionnaire_df["plan long journeys in advance"] = 6 - self.questionnaire_df["plan long journeys in advance"]

    @staticmethod
    def _convert_transform_str_to_int(x):
        ''' Converts the string transform data type into tuple(array, array) type.
        '''

        x = " ".join(x.replace("]", "[").split())
        bracket_split = x.split("[")

        position_str = bracket_split[2]
        rotation_str = bracket_split[4]

        positions = [float(i) for i in position_str.strip().split(" ")]
        rotations = [float(i) for i in rotation_str.strip().split(" ")]

        return (positions, rotations)

    
    def _vectorize_personalities(self):
        ''' Evaluates MDSI outcomes to vector format.
        '''

        df = self.questionnaire_df

        df['score_reckless'] = df[DISSOCIATIVE_COLUMNS].mean(axis=1)
        df['score_anxious'] = df[ANXIOUS_COLUMNS].mean(axis=1)
        df['score_risky'] = df[RISKY_COLUMNS].mean(axis=1)
        df['score_angry'] = df[ANGRY_COLUMNS].mean(axis=1)
        df['score_high_velocity'] = df[HIGH_VELOCITY_COLUMNS].mean(axis=1)
        df['score_distress_reduction'] = df[DISTRESS_REDUCTION_COLUMNS].mean(axis=1)
        df['score_patient'] = df[PATIENT_COLUMNS].mean(axis=1)
        df['score_careful'] = df[CAREFUL_COLUMNS].mean(axis=1)

        # print(self.questionnaire_df[MDSI_COLUMNS])


    def plot_personality_tsne(self, color_code="gender", dimension=2):
        '''
        Uses PCA and t-SNE to plot personality distribution in 2-D or 3-D space.
        '''

        assert 'score_reckless' in self.questionnaire_df.columns, "personality scores not computed"

        tsne = TSNE(dimension, perplexity=5)
        tsne_result = tsne.fit_transform(self.questionnaire_df[MDSI_COLUMNS])

        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'color_code': self.questionnaire_df[color_code]})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='color_code', data=tsne_result_df, ax=ax,s=120)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        plt.show()

    def get_scenario_data(self, scenario_num):
        scenario_data = [] 

        for data in self.sim_dfs: 
            if data.iloc[0]["scenario_num"] == scenario_num:
                scenario_data.append(data)

        return scenario_data

    def animate_trajectories(self, scenario=0):
        
        c = list(mcolors.TABLEAU_COLORS)

        dfs = self.get_scenario_data(scenario)

        dfs = [df[["frame_number", "transform_z", 
                 "normalized_pos_x", "normalized_pos_y", 
                 "normalized_rot_x", "user_id"]] for df in dfs]

        frames = min([len(df) for df in dfs])

        fig = plt.figure()
        ax = plt.axes()

        xs = [df["normalized_pos_x"] for df in dfs]
        ys = [df["normalized_pos_y"] for df in dfs]
        user_ids = [df.iloc[0]["user_id"] for df in dfs]

        def animate(i):
            ax.cla()

            for j,df in enumerate(dfs):
                x = xs[j]
                y = ys[j]

                line, = ax.plot(x[:i], y[:i], color=c[j])  # update the data.

                if i < frames - 1 and i > 20:
                    ax.annotate("", xy=(x[i],y[i]), xytext=(x[i-2],y[i-2]), arrowprops=dict(arrowstyle="->", color=line.get_color()))
                    ax.annotate("user_%s" % (user_ids[j]),
                        xy=(x[i],y[i]), 
                        xytext=(1,0), textcoords='offset points',
                        color=line.get_color()
                    )

            ax.set_xlim([np.min(x) - 10, np.max(x) + 10]) # fix the x axis
            ax.set_ylim([np.min(y) - 10, np.max(y) + 10]) # fix the y axis

            ax.set_xlabel('X position', 
               fontweight ='bold')
            ax.set_ylabel('Y position', 
               fontweight ='bold')
            
            # ax.set_zlim3d(-10, 10) # fix the y axis
            return line,

        ani = animation.FuncAnimation(fig, animate, frames = frames, interval=1, blit=True)

        # To save the animation, use e.g.
        #
        # ani.save("animation.gif")
        #
        # or
        #
        writer = animation.FFMpegWriter(
            fps=60, metadata=dict(artist='laura:)'), bitrate=1800)
        ani.save("scenario_%s.mp4" % (scenario), writer=writer)

        # plt.show()

if __name__ == "__main__":

    DATADIR = "./data"
    EXCLUSIONS = [2]

    stats = StatsManager(DATADIR, EXCLUSIONS)

    # stats.plot_personality_tsne(dimension=2)
    stats.animate_trajectories(scenario=0)

    # for df in stats.sim_dfs:
    #     print(df["transform"][0])

    # print(stats.questionnaire_df["gender"])