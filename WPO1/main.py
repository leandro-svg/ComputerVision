import numpy as np
import argparse
import cv2
import math
import time 
import tqdm
from matplotlib import pyplot as plt 
import os

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
 
        print(x, ' ', y)
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open('Inputs/image_coordinate.txt', 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image', image)

def click_event_left(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
 
        print(x, ' ', y)
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_left, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open('Inputs/image_coordinate_left.txt', 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image_left', image_left)
 

class Calibration():
    def __init__(self, ):
        super().__init__()
    def PreProcess(self, img_path):
        
        image = cv2.imread(img_path)
        return image

    def getPointsFromImage(self, image, path, choice):

        if not(os.path.isfile(path)):
            cv2.imshow('image', image)
            if (choice == 0):
                cv2.imshow('image', image)
                cv2.setMouseCallback('image', click_event)
            elif (choice == 1 ):
                cv2.imshow('image_left', image)
                cv2.setMouseCallback('image_left', click_event_left)
            cv2.waitKey(1000000)
            cv2.destroyAllWindows()
        else:
            print("Image Coordinate file already exists, we will work with it")
        with open(path, 'r') as file:
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
        
        
        ################################################
        inv_world_coord = np.linalg.pinv(world_coord.T)
        
        M = np.dot(image_coord.T, inv_world_coord)
        ################################################
        P = np.array([])
        for i in range(np.shape(image_coord)[0]):
            print(image_coord[i,:])
            print(world_coord[i,0:3])
            print(np.zeros((1,4))[0])
            first = np.concatenate((world_coord[i,:], np.zeros((1,4))[0], -1*image_coord[i,0]*world_coord[i,:]))
            second = np.concatenate((np.zeros((1,4))[0], world_coord[i,:],  -1*image_coord[i,1]*world_coord[i,:]))
            
            P = np.append(P, first, axis=0)
            P = np.append(P, second, axis=0)
            
        P = np.reshape(P, (24, 12))
        u, s, vh = np.linalg.svd(P)
        
        Mbis = vh[:, -1]
        Mbis = np.reshape(M, (4,3)).T
        M = Mbis
        ################################################
        
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
        extrinsic_add = np.concatenate((extrinsic, added))
        
        print("MMMMMM3", np.dot(intrinsic, extrinsic))
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
    
    def leftImage(self, image_left,path,  camParameters):
        image_points_left_temp = self.getPointsFromImage(image_left, "Inputs/image_coordinate_left.txt", 1)
        
        image_coord_left = np.array([])
        for elem in image_points_left_temp:
            elem = np.append(elem, [1])
            image_coord_left = np.append(image_coord_left, [elem])
        image_coord_left = np.reshape(image_coord_left, (np.shape(image_points_left_temp)[0],3))
        
        world2image = np.dot(camParameters["Intrinsic"],camParameters["Extrinsic"])
        M = np.zeros((3,3))
        for i in range(0,3):
            M[i,0] = world2image[i,0]
            M[i,1] = world2image[i,1]
            M[i,2] = world2image[i,3]
        print(M)
        image2world = np.linalg.inv(M)
        world_coord_left = np.dot(image2world,  image_coord_left.T)
        print("Those are world coordinate")
        print(world_coord_left)
    
    
    
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
            "--left",
            default="Inputs/left.jpg",
            type=str,
            help="Directory to left input images",
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
    img_path_left = args.left

    world_coord_file = args.txtfile
    Calibration = Calibration()
    
    image = Calibration.PreProcess(img_path_left)
    
    image_points = Calibration.getPointsFromImage(image, 'Inputs/image_coordinate.txt', 0)

    M, image_coord, world_coord = Calibration.projectMatrix(image_points, world_coord_file)
    
    camParameters = Calibration.getCameraParameters(M, image_coord, world_coord)
    
    image_left = Calibration.PreProcess(img_path_left)
    
    Calibration.leftImage(image_left,img_path_left,  camParameters)