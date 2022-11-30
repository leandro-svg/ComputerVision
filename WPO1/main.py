import numpy as np
import argparse
import cv2
import math
import time 
import tqdm
from matplotlib import pyplot as plt 
import os

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open('Inputs/image_coordinate.txt', 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image', image)
 

class Calibration():
    def __init__(self, ):
        super().__init__()
    def PreProcess(self, img_path):
        
        image = cv2.imread(img_path)
        return image

    def getPointsFromImage(self, image):

        if not(os.path.isfile("Inputs/image_coordinate.txt")):
            cv2.imshow('image', image)
            cv2.setMouseCallback('image', click_event)
            cv2.waitKey(1000000)
            cv2.destroyAllWindows()
        else:
            print("Image Coordinate file already exists, we will work with it")
        with open('Inputs/image_coordinate.txt', 'r') as file:
            lines = [line.rstrip() for line in file]

        image_coord = []
        for elem in lines: 
            for coord in elem.split(',') : 
                image_coord.append(int(coord))
        image_coord = np.reshape(image_coord, (np.size(lines),2))
        return image_coord

    def projectMatrix(self, image_coord_temp, world_coord_file):
        with open(world_coord_file, 'r') as file:
            world_lines = [line.rstrip() for line in file]
        while("" in world_lines):
            world_lines.remove("")

        world_coord_temp = []
        for elem in world_lines: 
            for coord in elem.split('   ') :
                world_coord_temp.append(int(coord))
        world_coord_temp = np.reshape(world_coord_temp, (np.size(world_lines),3))

        world_coord = np.array([])
        for elem in world_coord_temp:
            elem = np.append(elem, [1])
            world_coord = np.append(world_coord, [elem])
        world_coord = np.reshape(world_coord, (np.shape(world_coord_temp)[0],4))

        image_coord = np.array([])
        for elem in image_coord_temp:
            elem = np.append(elem, [1])
            image_coord = np.append(image_coord, [elem])
        image_coord = np.reshape(image_coord, (np.shape(image_coord_temp)[0],3))
        
        inv_world_coord = np.linalg.pinv(world_coord.T)
        
        M = np.dot(image_coord.T, inv_world_coord)
        return M, image_coord.T, world_coord.T

    def getCameraParameters(self, M, image_coord, world_coord):
        m_1  = M[0, 0:3]
        m_2 = M[1, 0:3]
        m_3 = M[2,0:3]
        r_3 = m_3
        c_x = np.dot(m_1, r_3)
        c_y = np.dot(m_2, r_3)
        
        f_x = np.linalg.norm(m_1*r_3)
        f_y = np.linalg.norm(m_2*r_3)
        
        r_1 = (1/f_x)*(m_1 - c_x*m_3)
        r_2 = (1/f_y)*(m_2 - c_y*m_3)
        
        t_x = (1/f_x)*(M[0,3] - c_x*M[2, 3])
        t_y = (1/f_y)*(M[1,3] - c_x*M[2, 3])
        t_z = M[2,3]

        R = np.reshape(np.concatenate((r_1, r_2, r_3)), (3,3))
        T = np.array([t_x, t_y, t_z])
        intrinsic  = np.array([[f_x, 1, c_x], [0, f_y, c_y], [0, 0, 1]])
        
        extrinsic = np.concatenate((R, np.array([T]).T), axis=1)
        added = np.array([[0,0,0,1]])
        extrinsic = np.concatenate((extrinsic, added))
        camParameters = {}
        
        camParameters["ImageCoordinate"] = image_coord
        camParameters["WorldCoordinate"] = world_coord
        camParameters["ProjectionMatrix"] = M
        camParameters["Intrinsic"] = intrinsic
        camParameters["Extrinsic"] = extrinsic
        camParameters["RotationMatrix"] = R
        camParameters["TranslationMatrix"] = T
        print(camParameters)
        return camParameters
def get_Parser():
    parser = argparse.ArgumentParser(
            description="Implementation of the required Calibration")
    parser.add_argument(
            "--input",
            default="Inputs/calibration_pointnumbering.jpg",
            type=str,
            help="Directory to input images",
            )

    parser.add_argument(
            "--txtfile",
            default="Inputs/calibration_points3.txt",
            type=str,
            help="text file where the woolrd coordinates are",
            )
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    img_path = args.input

    world_coord_file = args.txtfile
    Calibration = Calibration()
    
    image = Calibration.PreProcess(img_path)
    
    image_points = Calibration.getPointsFromImage(image)

    M, image_coord, world_coord = Calibration.projectMatrix(image_points, world_coord_file)
    
    camParameters = Calibration.getCameraParameters(M, image_coord, world_coord)
    