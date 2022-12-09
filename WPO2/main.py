import numpy as np
import argparse
import cv2
import cv2 as cv
import math
import time 
import tqdm
import os
import glob
from random import randint
from skimage.transform import warp, ProjectiveTransform
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plt.style.use('seaborn-poster')

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def click_event(event, x, y, flags, params):
    image3D, path = params
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image3D, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 1)

        with open(path, 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image', image3D)

class OpticalFlow():
    def __init__(self, ):
        super().__init__()
    
    def preProcess(self, image_path):
        print(image_path)
        if len(image_path) == 1:
            image_path = glob.glob(os.path.expanduser(image_path[0]))
            assert image_path, "The input path(s) was not found"
        for img_path in tqdm.tqdm(image_path):
            img = cv2.imread(img_path)
            real_image = img.copy()
            
        img1 = cv2.imread(image_path[4])
        img2 = cv2.imread(image_path[5])
        img_1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
        img_1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
        img_2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
        img_2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
        
        fx = img_1_x + img_2_x
        fy = img_1_y + img_2_y
        
        img1 = cv2.GaussianBlur(img1, (5, 5), 0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    
        
def get_Parser():
    parser = argparse.ArgumentParser(
            description="Implementation of the required Calibration")
    parser.add_argument(
        "--input",
        default="/home/leand/ULB_course/ComputerVision/WPO2/Basketball/frame1.png",
        nargs="+",
        help="A file or directory of your input data ",
        )
    
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input
    
    OF = OpticalFlow()
    OF.preProcess(img_path)