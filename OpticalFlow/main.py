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
        
    def computeFeature(self, image_path, index):
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
        It = signal.convolve2d(img2, kernel_t, boundary='symm', mode=mode)+ signal.convolve2d(img1, - kernel_t, boundary='symm', mode=mode)
        
        return Ix, Iy, It, img_1, img1, img2
    
    def Horn_Shunck_OF(self, fig, images, image_path):
        print(('Horn-Shunck Optical Flow'))
        print("Computing...")
        max_iter_list = [1, 1, 5, 10, 50, 100,500]
        for index in tqdm.tqdm(range(1,len(images))):            
            Ix, Iy, It, img_1, img1, img2 = self.computeFeature(image_path, index)
            
            avg_u,avg_v,u,v, iteration  = 0,0,0,0,0
            avg_u = u
            avg_v = v
            lamb = 20
            max_iter =1
            while iteration < max_iter:
                u = avg_u - (Ix*avg_u + Iy*avg_v + It)*Ix/(1+ lamb*(Ix ** 2 + Iy ** 2))
                v = avg_v - (Ix*avg_u + Iy*avg_v + It)*Iy/(1+ lamb*(Ix ** 2 + Iy ** 2))
                kernel = np.ones((3,3),np.float32)/9
                avg_u = cv.filter2D(u,-1,kernel)
                avg_v = cv.filter2D(v,-1,kernel)
                iteration += 1
                
            flow = flow_uv_to_colors(u,v)
            ax = fig.add_subplot(3, 3, index)
            
            ax.imshow(flow)
            ax.set_title("Horn-Shunck optical Flow with max_iter = %i" % max_iter)
            
        return flow
    
    def lucas_kanade_OF(self, fig, images, image_path):
        print('Lucas Kanade Optical Flow')
        print("Computing...")
        for index in tqdm.tqdm(range(1,len(images)-1)):            
            Ix, Iy, It, img_1, img1, img2 = self.computeFeature(image_path, index)
            
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
                    u[i, j] = uv[0]/30
                    v[i, j] = uv[1]/30
            n, m = u.shape
            flow = flow_uv_to_colors(u,v)
            ax6 = fig.add_subplot(3, 3, index)
      
            
            ax6.imshow(flow)
            ax6.set_title('Lucas Kanade Optical Flow')
            
        return flow
    
    
    def preProcess(self, image_path):
        print("Pre-processing images")
        if len(image_path) == 1:
            image_path = glob.glob(os.path.expanduser(image_path[0]))
            assert image_path, "The input path(s) was not found"
        images = []
        for img_path in tqdm.tqdm(image_path):
            img = cv2.imread(img_path)
            images.append(img)
            real_image = img.copy()
        return images

    def main(self, images, image_path ):
        fig = plt.figure(figsize= (15, 15))
        fig6 = plt.figure(figsize= (15, 15))
        
        Horn_Shunck_OF = self.Horn_Shunck_OF(fig,images, image_path)
        fig.suptitle("Horn-Shunck Optical Flow", fontsize=15)
        
        Lucas_Kanade_OF = self.lucas_kanade_OF(fig6,  images, image_path)
        fig6.suptitle("Lucas Kanade Optical Flow", fontsize=15)
        
        fig.savefig('output/Horn_Shunck_OF_demoframes.png')
        fig6.savefig('output/lucas_kanade_OF_demoframes.png')
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
    OF = OpticalFlow()
    images = OF.preProcess(img_path)
    OF.main(images, img_path)