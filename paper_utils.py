import os, sys, glob 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2

def stitch_scenarios(dir):
    
    lbls = ['Scenario 1: Jaywalking Pedestrian', 
    'Scenario 2: Static Obstacles on Highway', 
    'Scenario 3: Yellow Light Interaction', 
    'Scenario 4: Control']

    font = cv2.FONT_HERSHEY_SIMPLEX

    images = glob.glob(os.path.join(dir, "*.png"))
    images = sorted(images)

    images_np = []

    for i, image in enumerate(images): 
        im = cv2.imread(image)
        h,w = im.shape[0], im.shape[1]

        im = im[150:(h-150), 200:(w-200)]
        cv2.putText(im, lbls[i], (200, 200), font, 3, (0, 0, 0), 10)

        w = int(im.shape[1] * 40 / 100)
        h = int(im.shape[0] * 40 / 100)
        im = cv2.resize(im, (w,h))
    
        images_np.append(im)
    
    im_h = cv2.hconcat(images_np)

    cv2.imwrite("figs/scenarios_stitched.png", im_h)

if __name__ == "__main__":
    stitch_scenarios("./figs/scenario_screenshots")
