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
from scipy import signal
from PIL import Image

from FrankotChellappa import frankotchellappa

class PhotometricStereo():
    def __init__(self, ):
        super().__init__()
        
    def calculate_brightness(self, image):
        greyscale_image = image.convert('L')
        histogram = greyscale_image.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale   

    def get_mask(self, img):
        tresh = 0.1
        mask = np.zeros((np.shape(img[0])[0]))
        sum_img = sum(img)/np.shape(img)[0]/ 255
        mask[sum_img >= tresh] = 1
        mask[sum_img < tresh] = 0
        return mask

    def main(self,image_path, file_path):
        """
        source : 
        https://www.cs.cornell.edu/courses/cs4670/2018sp/lec25-photometric-stereo.pdf
        """
        print("Pre-processing Step")
        if len(image_path) == 1:
            image_path = glob.glob(os.path.expanduser(image_path[0]))
            assert image_path, "The input path(s) was not found"
        I = np.zeros((len(image_path), 850*450))
        for index, img_path in tqdm.tqdm(enumerate(image_path)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            vector_img = np.reshape(img, (850*450))
            I[index, :] = vector_img
        mask  = self.get_mask(I)
        with open(file_path, "r") as file:
            lines = file.readlines()
            points = np.loadtxt(lines)
        L = points

        I_mask = I*mask
        L_pinv = np.linalg.pinv(L.T)
        G = np.matmul(L_pinv, I_mask)

        p = G[0,:] / G[2,:]
        q = G[1,:] / G[2,:]
        
        p[np.isnan(p)] = 0
        q[np.isnan(q)] = 0

        p  = np.reshape(p, np.shape(img))
        q = np.reshape(q, np.shape(img))

        gradient_space = np.zeros([p.shape[0], p.shape[1], 3])
        gradient_space[:, :, 0] = p
        gradient_space[:, :, 1] = q
        gradient_space[:, :, 2] = -1
        cv2.imwrite("gradient_space.jpg", gradient_space*255)
        
        depth = frankotchellappa(p, q)
        """ 
        * Results are Complex Numbers

        Again due to the use of DFT's, the results are complex numbers.
        In principle an ideal gradient field of real numbers results
        a real-only result. This "imaginary noise" is observerd
        even with theoretical functions, which leads to the conclusion
        that it is due to a numerical noise. It is left to the user
        to decide what to do with noise, for instance to use the
        modulus or the real part of the result. But it
        is recomended to use the real part.
        """
        depth = depth.real
        plt.imshow(depth)
        plt.savefig("Depthmap.jpg")
        
 
    


        

    
def get_Parser():
    parser = argparse.ArgumentParser(
            description="Implementation of the required Calibration")
    parser.add_argument(
        "--input",
        default=".",
        nargs="+",
        help="A file or directory of your input dat",
        )
    parser.add_argument(
        "--light",
        default=".",
        nargs="+",
        help="Light input text file",
        )
    
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input
    file_path = args.light
    print(img_path)
    PS = PhotometricStereo()
    PS.main(img_path, file_path[0])