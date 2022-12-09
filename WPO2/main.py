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
from view_flow import flow_uv_to_colors
from scipy import signal

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
            
        images = []
        for img_path in tqdm.tqdm(image_path):
            img = cv2.imread(img_path)
            images.append(img)
            real_image = img.copy()
            
        for index in range(1,len(images)):

            img1 = cv2.imread(image_path[index-1], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(image_path[index], cv2.IMREAD_GRAYSCALE)
            img1 = img1/255
            img2 = img2/255
            img_1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
            img_1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
            img_2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
            img_2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
            
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)
            
            kernel_t = np.array([[1., 1.], [1., 1.]])
            Ix =  img_1_x # + img_2_x
            Iy = img_1_y # + img_2_y
            # It = img1 - img2 
            mode = 'same'
            It = signal.convolve2d(img2, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(img1, - kernel_t, boundary='symm', mode=mode)
            avg_u,avg_v,u,v, iteration  = 0,0,0,0,0
            avg_u = u
            avg_v = v
            while iteration < 1e2:
                
                u = avg_u - (Ix*avg_u + Iy*avg_v + It)*Ix/(1+ 1*(Ix ** 2 + Iy ** 2))
                v = avg_v - (Ix*avg_u + Iy*avg_v + It)*Iy/(1+ 1*(Ix ** 2 + Iy ** 2))
                kernel = np.ones((2,2),np.float32)/25
                avg_u = cv.filter2D(u,-1,kernel)
                avg_v = cv.filter2D(v,-1,kernel)
                
                iteration += 1
                
            flow = flow_uv_to_colors(u,v)
            plt.imshow(flow)
            plt.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            
       

    
        
def get_Parser():
    parser = argparse.ArgumentParser(
            description="Implementation of the required Calibration")
    parser.add_argument(
        "--input",
        default="/home/leand/ULB_course/ComputerVision/WPO2/Basketball/frame1.png",
        nargs="+",
        help="A file or directory of your input dat",
        )
    
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input
    
    OF = OpticalFlow()
    OF.preProcess(img_path)