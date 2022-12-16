import numpy as np
import argparse
import cv2
import cv2 as cv
import tqdm
import os
import glob
import time
from random import randint
from skimage.transform import warp, ProjectiveTransform
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from view_flow import flow_uv_to_colors
from scipy import signal

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
        fig = plt.figure(figsize= (10, 10))
        fig2 = plt.figure(figsize= (10, 10))
        fig3 = plt.figure(figsize= (10, 10))
        fig4 = plt.figure(figsize= (10, 10))
        fig5 = plt.figure(figsize= (10, 10))
        fig6 = plt.figure(figsize= (10, 10))
        fig7 = plt.figure(figsize= (10, 10))

        for index in range(1,len(images)):
            print("Image :", index)
            img_1 = cv2.imread(image_path[index-1], cv2.IMREAD_GRAYSCALE)
            img_2 = cv2.imread(image_path[index], cv2.IMREAD_GRAYSCALE)
            img1 = img_1#/255
            img2 = img_2#/255
            img_1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
            filter_x = np.transpose(np.array([[-1., -1.], [1., 1.]]))
            
            img_1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
            img_2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)
            img_2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)
            
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)
            
            kernel_t = np.array([[1., 1.], [1., 1.]])
            Ix =  img_1_x + img_2_x
            Iy = img_1_y + img_2_y
            
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
            img_flo = np.concatenate([img_1, flow[:,:,1]], axis=0)
            
            ax = fig.add_subplot(3, 3, index)
            ax.imshow(flow)
            ax2 = fig2.add_subplot(3, 3, index)
            ax2.imshow(v)
            ax3 = fig3.add_subplot(3, 3, index)
            ax3.imshow(u)

            u = np.zeros(img1.shape)
            v = np.zeros(img1.shape)
            n = 1
            for i in range(1, u.shape[0]):
                for j in range(1, u.shape[1]):
                    Ixpi = np.sum(np.power(Ix[i - n:i+n+1, j - n:j + n + 1],2))
                    Iypi = np.sum(np.power(Iy[i - n:i+n+1, j - n:j + n + 1],2))
                    IxIy = np.sum(np.multiply(Ix[i - n:i+n+1, j - n:j + n + 1], Iy[i - n:i+n+1, j - n:j + n + 1]))
                    A = np.array([[Ixpi, IxIy],[IxIy,Iypi]])
                    try:
                        A_inv = np.linalg.inv(A)
                    except:
                        A_inv = A
                    IxIt = -1*np.sum(np.multiply(Ix[i - n:i+n+1, j - n:j + n + 1], It[i - n:i+n+1, j - n:j + n + 1]))
                    IyIt = -1*np.sum(np.multiply(Iy[i - n:i+n+1, j - n:j + n + 1], It[i - n:i+n+1, j - n:j + n + 1]))
                    P = np.array([[IxIt],[IyIt]])
                    uv = np.matmul(A_inv, P)
                    u[i, j] = uv[0]
                    v[i, j] = uv[1]
            n, m = u.shape
            [X,Y] = np.meshgrid(np.arange(m, dtype = 'float64'), np.arange(n, dtype = 'float64'))
            new_flow = flow_uv_to_colors(u,v)
            ax4 = fig4.add_subplot(3, 3, index)
            ax4.imshow(u)
            ax5 = fig5.add_subplot(3, 3, index)
            ax5.imshow(v)
            ax6 = fig6.add_subplot(3, 3, index)
            ax6.imshow(new_flow)
            img_flo = np.concatenate([img_flo, new_flow[:,:,1]], axis=0)
            cv2.imshow('image', img_flo[:, :]/255.0)
            cv2.waitKey()
        
        plt.show()
         
    
def get_Parser():
    parser = argparse.ArgumentParser(
            description="Implementation of the required Calibration")
    parser.add_argument(
        "--input",
        default="./Basketball/*",
        nargs="+",
        help="A file or directory of your input dat",
        )
    
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input
    print(img_path)
    OF = OpticalFlow()
    OF.preProcess(img_path)