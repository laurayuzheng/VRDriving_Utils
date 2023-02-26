''' 
Run statistics on collected CSV data. 
Created by Laura Zheng 
Feb 16, 2023
'''

import sys 
import os, glob
import itertools
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LinearSegmentedColormap

from config import * 

import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.manifold import TSNE

SCENARIO_DESCRIPTIONS = {
    0: "Jaywalking pedestrian", 
    1: "Static obstacles on highway",
    2: "Yellow light on intersection approach",
    3: "Control"
}

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
    # "attempt to drive away from traffic lights in third gear (or on the neutral mode in automatic cars)",
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
        self.baseline_sim_dfs = None 

        self._get_data()
        self._vectorize_personalities()

    def print_sim_columns(self):
        for i,name in enumerate(self.sim_dfs[0].columns.to_list()):
            print(i, name)

    def print_questionnaire_columns(self):
        for i,name in enumerate(self.questionnaire_df.columns.to_list()):
            print(i, name)

    def get_personality_data(self):
        return self.questionnaire_df[MDSI_COLUMNS].drop(index=EXCLUSIONS).to_numpy()
    
    def get_sim_data(self, scenario, user=None):

        data = []
        
        stuff_to_look_through = self.baseline_sim_dfs if user == -1 else self.sim_dfs

        for df in stuff_to_look_through:
            
            # only want relevant scenario data
            if df.iloc[0]["scenario_num"] != scenario:
                continue 

            if user is not None and (df.iloc[0]["user_id"] != user or df.iloc[0]["user_id"] in EXCLUSIONS):
                continue 

            # if not looking for specific user, exclude the baseline
            # if user is None and df.iloc[0]["user_id"] == 0:
            #     continue 

            user_data = []
            length = 0
            last_t = (0,0)
            for i, row in df.iterrows():
                t = (row["normalized_pos_x"], row["normalized_pos_x"])
                # t = (row["transform_x"], row["transform_x"])
                user_data.append(t)
                last_t = t
                length = i
            
            # Zero padding the trajectory
            for i in range(0, MAX_STEPS-length-1):
                user_data.append(last_t)

            assert len(user_data) == MAX_STEPS, "user data not the right length. Actual length: %d" % (len(user_data))

            data.append(user_data)

        data = np.array(data)
        data -= np.amin(data)
        data /= (np.amax(data) - np.amin(data))

        return data
            

    def _get_data(self):
        '''
        Gets data from self.datadir and updates self.questionnaire_df and self.sim_df.
        '''

        sim_data_csv_list = glob.glob(os.path.join(self.datadir, "sim_data/*/*.csv"))

        self.sim_dfs = [] 
        self.baseline_sim_dfs = []

        for filename in sim_data_csv_list:

            if "CARLAROUTE" in filename:
                continue 

            exp_id = "_".join(filename.split("/")[-2].split("_")[:3])
            user_id = int(exp_id.split("_")[0]) - 2
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
            first_rot_z = df["transform"].iloc[20][1][2]

            df["transform_x"] = df["transform"].apply(lambda x: x[0][0] - first_x)
            df["transform_y"] = df["transform"].apply(lambda x: x[0][1] - first_y)
            df["transform_z"] = df["transform"].apply(lambda x: x[0][2] - first_z)

            df["rotation_x"] = df["transform"].apply(lambda x: x[1][0])
            df["rotation_y"] = df["transform"].apply(lambda x: x[1][1])
            df["rotation_z"] = df["transform"].apply(lambda x: x[1][2])

            # Rotate trajectories so always driving forward 

            if scenario_num == 2: 
                rot_matrix = rotation_matrix(90 - first_rot_z)
            else:
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

            df = df[ df['normalized_pos_y'] < 250 ]

            if user_id == -1:
                self.baseline_sim_dfs.append(df)
            else:
                self.sim_dfs.append(df)

        # Sort based on user id (primary) and scenario number (secondary)
        self.sim_dfs = sorted(self.sim_dfs, key=lambda x: x.iloc[0]["user_id"]+0.1*x.iloc[0]["scenario_num"])

        self.baseline_sim_dfs = sorted(self.baseline_sim_dfs, key=lambda x: x.iloc[0]["scenario_num"])

        # assert self.sim_dfs[0].iloc[0]["user_id"] == -1, "baseline doesn't exist"

        # self.baseline_sim_df = self.sim_dfs.pop(0)

        self._process_questionnaire_df()


    def _process_questionnaire_df(self):
        self.questionnaire_df = pd.read_csv(os.path.join(self.datadir, "questionnaire.csv"), index_col=None, header=0)
        self.questionnaire_df['user_id'] = self.questionnaire_df.index
        
        self.questionnaire_df.columns = self.questionnaire_df.columns.str.replace(".1", "", regex=True).str.strip()
        self.questionnaire_df.rename({'What is your gender?': 'gender'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'Full Name': 'name'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'What is your ethnicity?': 'ethnicity'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'Select your age group': 'age_group'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'Select your role': 'role'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'Have you experienced an autonomous vehicle before (e.g., Tesla\'s autopilot)?': 'autonomous_vehicle_experience'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'What is your ethnicity?': 'ethnicity'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'What kind of driver would you classify yourself as?': 'personality_self_classification'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'How do you get to school?': 'commute_method'}, axis=1, inplace=True)
        self.questionnaire_df.rename({'Do you have a drivers license?': 'has_drivers_license'}, axis=1, inplace=True)

        self.questionnaire_df.rename(columns={self.questionnaire_df.columns[10]: 'discomfort',
                                              self.questionnaire_df.columns[11]: 'fatigue', 
                                              self.questionnaire_df.columns[12]: 'headache',
                                              self.questionnaire_df.columns[13]: 'eyestrain', 
                                              self.questionnaire_df.columns[14]: 'difficulty_focusing', 
                                              self.questionnaire_df.columns[15]: 'salivation',
                                              self.questionnaire_df.columns[16]: 'sweating',
                                              self.questionnaire_df.columns[17]: 'nausea',
                                              self.questionnaire_df.columns[18]: 'difficulty_concentrating', 
                                              self.questionnaire_df.columns[19]: 'fullness_of_head', 
                                              self.questionnaire_df.columns[20]: 'blurred_vision', 
                                              self.questionnaire_df.columns[21]: 'dizziness_eyes_open', 
                                              self.questionnaire_df.columns[22]: 'dizziness_eyes_closed', 
                                              self.questionnaire_df.columns[23]: 'vertigo', 
                                              self.questionnaire_df.columns[24]: 'stomach_awareness', 
                                              self.questionnaire_df.columns[25]: 'burping'
                                            }, inplace=True)

        
        # Invert the scores for some questions for analysis (based on original MDSI paper)
        self.questionnaire_df["feel I have control over driving"] = 6 - self.questionnaire_df["feel I have control over driving"]
        self.questionnaire_df["feel comfortable while driving"] = 6 - self.questionnaire_df["feel comfortable while driving"]
        self.questionnaire_df["when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves"] = 6 - self.questionnaire_df["when a traffic light turns green and the car in front of me doesn’t get going, I just wait for a while until it moves"]
        self.questionnaire_df["plan long journeys in advance"] = 6 - self.questionnaire_df["plan long journeys in advance"]

        self.questionnaire_df.columns = [name if duplicated == False else name + "_2" for duplicated, name in zip(self.questionnaire_df.columns.duplicated(), self.questionnaire_df.columns)]

        self.questionnaire_df = self.questionnaire_df.drop(['blow my horn or “flash” the car in front as a way of expressing frustrations_2', 
                                                            "like to take risks while driving_2",
                                                            "meditate while driving_2",  
                                                            'Please select "6" for this question.',
                                                            ], axis=1)

        self.questionnaire_df.rename(columns={
                                        "How easy was it for you to adjust to the simulated driving?": 'vr_ease_adjustment',
                                        "From a scale from to0, do you feel as though the scenarios were realistic?": 'vr_realistic', 
                                        'Please rate your sense of being in the virtual environment, on the following scale from to 7, where 7 represents your normal experience of being in a place. I had a sense of “being there” in the virtual environment:': 'vr_presence', 
                                        "To what extent were there times during the experience when the virtual environment was the reality for you?": 'vr_immersion', 
                                    }, inplace=True)
        
        
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
        
        # normalize scores for each participant
        df[MDSI_COLUMNS] = df[MDSI_COLUMNS].div(df[MDSI_COLUMNS].sum(axis=1), axis=0)

        # print(self.questionnaire_df[MDSI_COLUMNS])

    def save_to_csv(self, savedir="csv_output/"):
        os.makedirs(os.path.join(savedir, "simdata"), exist_ok=True)
        df = self.questionnaire_df
        df = df.drop(["name"], axis=1)
        df.to_csv(os.path.join(savedir,"questionnaire_processed.csv"), encoding='utf-8', index=False)

        for df in self.sim_dfs:
            df.to_csv(os.path.join(savedir,"simdata","%d_scenario_%d.csv"%(df["user_id"][0], df["scenario_num"][0])), encoding='utf-8', index=False)

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


    def plot_personality_style(self, color_code="gender", style1="score_risky", style2="score_anxious"):
        '''
        Plots two driving style values against each other, style1 being on x-axis and style2 being on y-axis.
        '''

        assert 'score_reckless' in self.questionnaire_df.columns, "personality scores not computed"



        tsne_result_df = pd.DataFrame({style1: self.questionnaire_df[style1], style2: self.questionnaire_df[style2], 'color_code': self.questionnaire_df[color_code]})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x=style1, y=style2, hue='color_code', data=tsne_result_df, ax=ax,s=120, legend=True)
        plt.legend(loc='upper right')
        
        lim = (0, 7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        plt.axhline(3.5, color='black')
        plt.axvline(3.5, color='black')
        ax.spines['bottom'].set_color('gray')
        ax.spines['top'].set_color('gray') 
        ax.spines['right'].set_color('gray')
        ax.spines['left'].set_color('gray')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

        plt.savefig("./figs/%s_vs_%s.pdf" % (style1, style2), bbox_inches='tight')

    def get_scenario_data(self, scenario_num):
        scenario_data = [] 
        indices = []

        for i, data in enumerate(self.sim_dfs): 
            if data.iloc[0]["scenario_num"] == scenario_num:
                scenario_data.append(data)
                indices.append(i)

        return scenario_data, indices

    def trajectory_heatmap(self, alpha_weights, scenario=0):
        
        baseline_data = self.baseline_sim_dfs[scenario]
        baseline_data = np.array([baseline_data["normalized_pos_x"].to_numpy(), baseline_data["normalized_pos_y"].to_numpy()])
        
        # print(baseline_data.shape)
        # print(self.baseline_sim_dfs[scenario].shape)

        def find_nearest_x(y):
            '''Find x point on baseline data closest to y value'''

            idx = np.abs(baseline_data[1] - y).argmin()
            return baseline_data[0][idx]

        # c = list(mcolors.TABLEAU_COLORS)

        dfs, _ = self.get_scenario_data(scenario)
        # quantile = self.questionnaire_df.score_risky.quantile(0.75)
        # users = self.questionnaire_df.index[self.questionnaire_df['score_risky'] >= quantile].tolist()

        dfs = [df[["frame_number", "transform_z", 
                 "normalized_pos_x", "normalized_pos_y", 
                 "normalized_rot_x", "user_id"]] for df in dfs]

        # frames = int(np.max([len(df.index) for df in dfs]))

        fig = plt.figure()
        ax = plt.axes()

        xs = [df["normalized_pos_x"] for df in dfs]
        ys = [df["normalized_pos_y"] for df in dfs]
        # user_ids = [df.iloc[0]["user_id"] for df in dfs]
        # max_risk = self.questionnaire_df['score_risky'].max()
        # user_color_alphas = [df.iloc[0]["user_id"] for df in dfs]
        
        for j,df in enumerate(dfs):
            
            # only graph risky high scorers
            # if j not in users:
            #     continue 

            x = xs[j].to_numpy()
            y = ys[j].to_numpy()

            if scenario != 1:
                for i in range(len(x)): 
                    x_baseline = find_nearest_x(y[i])
                    x[i] = x[i] - x_baseline

            colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
            cm = LinearSegmentedColormap.from_list(
                    "Custom", colors, N=5)
            # mat = np.indices((10,10))[1]
            alphas = alpha_weights[j][:len(x)]

            ax.scatter(x, y, color=cm(alphas), s=6, alpha=alphas)  # update the data.
            
            if scenario == 0:
                ax.axhline(y=42, color='blue', linestyle='dotted')
                ax.annotate("pedestrian",
                        xy=(-10,42), 
                        xytext=(0,2), textcoords='offset points',
                        color='blue',
                        size=5
                    )
            if scenario == 2:
                ax.axhline(y=55, color='blue', linestyle='dotted')
                ax.annotate("yellow traffic light",
                        xy=(-10,55), 
                        xytext=(0,2), textcoords='offset points',
                        color='blue',
                        size=5
                    )
        
        if scenario == 2:
            ax.set_xlim([np.min(x) - 30, np.max(x) + 30]) # fix the x axis
        else:
            ax.set_xlim([np.min(x) - 10, np.max(x) + 10]) # fix the x axis

        ax.set_ylim([np.min(y) - 10, np.max(y) + 10]) # fix the y axis

        ax.set_xlabel('X position', 
            fontweight ='bold')
        ax.set_ylabel('Y position', 
            fontweight ='bold')

        ax.set_title("Scenario %d: %s" % (scenario, SCENARIO_DESCRIPTIONS[scenario]))

        plt.savefig("figs/trajectory_heatmap_scenario%d.pdf" % (scenario))
        plt.close()


    def animate_trajectories(self, scenario=0):
        
        c = list(mcolors.TABLEAU_COLORS)

        dfs, _ = self.get_scenario_data(scenario)

        dfs = [df[["frame_number", "transform_z", 
                 "normalized_pos_x", "normalized_pos_y", 
                 "normalized_rot_x", "user_id"]] for df in dfs]

        frames = int(np.max([len(df.index) for df in dfs]))

        fig = plt.figure()
        ax = plt.axes()

        xs = [df["normalized_pos_x"] for df in dfs]
        ys = [df["normalized_pos_y"] for df in dfs]
        user_ids = [df.iloc[0]["user_id"] for df in dfs]
        max_risk = self.questionnaire_df['score_risky'].max()
        user_color_alphas = [df.iloc[0]["user_id"] for df in dfs]
        
        def animate(i):
            ax.cla()
            orig_i = i 

            for j,df in enumerate(dfs):
                x = xs[j]
                y = ys[j]
                
                # alphas = alpha_weights[j]

                i = min(orig_i, len(x)-1)

                color = c[j%len(c)] if j != 0 else 'black'

                line, = ax.plot(x[:i], y[:i], color=color, s=11)  # update the data.
                
                if scenario == 0:
                    ax.axhline(y=42, color='r', linestyle='dotted')
                    ax.annotate("pedestrian",
                            xy=(-10,42), 
                            xytext=(0,2), textcoords='offset points',
                            color='r',
                            size=5
                        )
                if scenario == 2:
                    ax.axhline(y=55, color='r', linestyle='dotted')
                    ax.annotate("yellow traffic light",
                            xy=(-10,55), 
                            xytext=(0,2), textcoords='offset points',
                            color='r',
                            size=5
                        )

                if i < frames - 1 and i > 20:
                    ax.annotate("", xy=(x[i],y[i]), xytext=(x[i-2],y[i-2]), arrowprops=dict(arrowstyle="->", color=line.get_color()))
                    
                    # if j == 0:
                    #     ax.annotate("baseline",
                    #         xy=(x[i],y[i]), 
                    #         xytext=(1,0), textcoords='offset points',
                    #         color=line.get_color()
                    #     )

                text_box = AnchoredText("Frame: %d" % (i), frameon=True, loc=4, pad=0.5)
                plt.setp(text_box.patch, facecolor='white', alpha=0.5)
                plt.gca().add_artist(text_box)

            ax.set_xlim([np.min(x) - 10, np.max(x) + 10]) # fix the x axis
            ax.set_ylim([np.min(y) - 10, np.max(y) + 10]) # fix the y axis

            ax.set_xlabel('X position', 
               fontweight ='bold')
            ax.set_ylabel('Y position', 
               fontweight ='bold')
            
            ax.set_title("Scenario %d: %s" % (scenario, SCENARIO_DESCRIPTIONS[scenario]))
            
            # ax.set_zlim3d(-10, 10) # fix the y axis
            return line,

        ani = animation.FuncAnimation(fig, animate, frames = frames, interval=1, blit=True)

        writer = animation.FFMpegWriter(
            fps=60, metadata=dict(artist='laura:)'), bitrate=1800)
        ani.save("./vids/scenario_%s.mp4" % (scenario), writer=writer)

        # plt.show()

if __name__ == "__main__":

    # MDSI_COLUMNS = [
    #     'score_reckless', 
    #     'score_anxious',
    #     'score_risky',
    #     'score_angry',
    #     'score_high_velocity',
    #     'score_distress_reduction',
    #     'score_patient',
    #     'score_careful'
    # ]

    # DATADIR = "./data"
    # EXCLUSIONS = [0]

    stats = StatsManager(DATADIR, EXCLUSIONS)

    stats.save_to_csv()
    # stats.print_sim_columns()
    # stats.print_questionnaire_columns()

    # stats.plot_personality_tsne(dimension=2)
    # stats.plot_personality_style()
    # stats.animate_trajectories(scenario=0)
    # stats.animate_trajectories(scenario=1)
    # stats.animate_trajectories(scenario=2)
    # stats.animate_trajectories(scenario=3)

    # for df in stats.sim_dfs:
    #     print(df.columns)

    # print(stats.questionnaire_df["gender"])