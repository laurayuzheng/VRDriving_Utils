import sys, os, glob
import numpy as np
import pandas as pd
from scipy.stats import kstest, ks_2samp, pearsonr, spearmanr

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

METRIC_COLUMNS = [
    # "steering_sum", 
    # "throttle_sum", 
    # "brake_sum",
    # "sinuosity",
    # "turning_angle_curvature",
    # "length_variation_curvature",
    # "steiner_formula_curvature",
    # "osculating_circle_curvature", 
    "steer_total", 
    "throttle_total", 
    "brake_total",
]

class IrosOrDieros: 
    '''
    Replicate stats computations for IROS VR Driving.
    '''

    def __init__(self, datadir):
        self.datadir = datadir 
        self.df = pd.read_csv(os.path.join(datadir, "questionnaire_processed.csv"), index_col=None, header=0)
        self.df = self.df.drop(index=[0, 17]) # first data point has no trajectory data :( 
    
    def get_quantile_group(self, column):
        quantiles = np.percentile(self.df[column], [25, 50, 75])

        first_quantile = self.df[self.df[column]<quantiles[0]]
        second_quantile = self.df[(self.df[column]>quantiles[0]) & (self.df[column]<quantiles[2])]
        third_quantile = self.df[self.df[column]>quantiles[2]]

        # return bottom 25%, middle 25%, and top 25% of df
        return first_quantile, second_quantile, third_quantile
    
    def ks_test(self, group1, group2, column):

        # sinuosity ks test between two groups
        ks_test = ks_2samp(group1[column], group2[column])
        print(ks_test)
    
    def pearson_test(self, column1, column2):

        # sinuosity ks test between two groups
        pearson = pearsonr(self.df[column1], self.df[column2])
        return pearson 

    def spearman_test(self, column1, column2):
        spearman = spearmanr(self.df[column1], self.df[column2])
        return spearman
        # print(spearman)

    def run_all_ks_tests(self):

        # Run ks tests within score columns
        
        for column in MDSI_COLUMNS:
            first,second,third = self.get_quantile_group(column)
            for metric in METRIC_COLUMNS:
                print("KS Test. \nGroup: first and third quantiles of %s \nMetric: %s" % (column, metric))
                self.ks_test(first, third, metric)
                print()

    # def plot_histogram(self, name, column):

    def run_all_pearsonr(self, scenario):
        significant_p_count = 0

        for column in MDSI_COLUMNS:
            statistics_contents = [] 
            pval_contents = []
                
            for metric in METRIC_COLUMNS:

                if scenario:
                    metric += "_%d" % (scenario)
                
                test = self.pearson_test(column, metric)
                statistics_contents.append(str(round(test.statistic, 3)))
                pval_contents.append(str(round(test.pvalue, 3)))
                
                if test.pvalue <= 0.05:
                    significant_p_count += 1

            print("Column: %s" % (column))
            print(" & ".join(statistics_contents))
            print(" & ".join(pval_contents))
            print()
        
        return significant_p_count 

    def run_all_spearmanr(self, scenario):
        significant_p_count = 0
            
        for column in MDSI_COLUMNS:
            statistics_contents = [] 
            pval_contents = []
            for metric in METRIC_COLUMNS:

                if scenario:
                    metric += "_%d" % (scenario)
                
                test = self.spearman_test(column, metric)
                statistics_contents.append(str(round(test.statistic, 3)))
                pval_contents.append(str(round(test.pvalue, 3)))

                if test.pvalue <= 0.05:
                    significant_p_count += 1
            
            print("Column: %s" % (column))
            print(" & ".join(statistics_contents))
            print(" & ".join(pval_contents))
            print()

        return significant_p_count 

    
if __name__ == "__main__":

    tester = IrosOrDieros("./csv_output")
    # tester.run_all_ks_tests()

    scenario = None
    ps = tester.run_all_pearsonr(scenario)
    print("pearson done. significant p count: %d \n" % (ps))

    ps = tester.run_all_spearmanr(scenario)
    print("spearman done. significant p count: %d \n" % (ps))


    #######
    
        
    # count_pearson = []
    # count_spearman = []
        
    # for scenario in [0,1,2,3]:
    #     ps = tester.run_all_pearsonr(scenario)
    #     print("pearson done. significant p count: %d \n" % (ps))
    #     count_pearson.append(ps)

    #     ps = tester.run_all_spearmanr(scenario)
    #     print("spearman done. significant p count: %d \n" % (ps))
    #     count_spearman.append(ps)

    # print(count_pearson)
    # print(count_spearman)