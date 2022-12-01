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
        cv2.putText(image_left, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open('Inputs/image_coordinate.txt', 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image_left', image_left)

def click_event_right(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
 
        print(x, ' ', y)
 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_right, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open('Inputs/image_coordinate_right.txt', 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image_right', image_right)
 

class Calibration():
    def __init__(self, ):
        super().__init__()
    def PreProcess(self, img_path):
        
        image = cv2.imread(img_path)
        return image

    def getPointsFromImage(self, image, path, choice):
        if not(os.path.isfile(path)):
            
            if (choice == 0):
                cv2.imshow('image_left', image)
                cv2.setMouseCallback('image_left', click_event)
            elif (choice == 1 ):
                cv2.imshow('image_right', image)
                cv2.setMouseCallback('image_right', click_event_right)
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
        # inv_world_coord = np.linalg.pinv(world_coord.T)
        
        # M = np.dot(image_coord.T, inv_world_coord)
        ################################################
        P = np.array([])
        for i in range(np.shape(image_coord)[0]):
            first = np.concatenate((world_coord[i,:], np.zeros((1,4))[0], -1*image_coord[i,0]*world_coord[i,:]))
            
            second = np.concatenate((np.zeros((1,4))[0], world_coord[i,:],  -1*image_coord[i,1]*world_coord[i,:]))
            P = np.append(P, first, axis=0)
            P = np.append(P, second, axis=0)
            
        P = np.reshape(P, (24, 12))
        # # print(P)
        # u, s, vh = np.linalg.svd(P)
        # print("S", s)
        # Mbis = vh[:, -1]
        
        # Mbis = np.reshape(Mbis, (4,3)).T
        # norm_3 = np.linalg.norm(Mbis[2,:])
        # M = Mbis/norm_3
        ################################################
        A = P
        A_ = np.matmul(A.T, A)
        # compute its eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(A_)
        
        # find the eigenvector with the minimum eigenvalue
        # (numpy already returns sorted eigenvectors wrt their eigenvalues)
        m = eigenvectors[:, 11]

        # reshape m back to a matrix
        M = m.reshape(3, 4)
        norm_3 = np.linalg.norm(M[2,:])
        M = M/norm_3
        
        return M, image_coord.T, world_coord.T

    def getCameraParameters(self, M, image_coord, world_coord):
        m_1  = M[0, 0:3]
        m_2 = M[1, 0:3]
        m_3 = M[2,0:3]
        r_3 = m_3
        c_x = np.matmul(m_1, r_3)
        c_y = np.matmul(m_2, r_3)
        
        f_x = np.linalg.norm(m_1*r_3)
        f_y = np.linalg.norm(m_2*r_3)
        
        r_1 = (1/f_x)*(m_1 - c_x*m_3)
        r_2 = (1/f_y)*(m_2 - c_y*m_3)
        
        t_x = (1/f_x)*(M[0,3] - c_x*M[2, 3])
        t_y = (1/f_y)*(M[1,3] - c_y*M[2, 3])
        t_z = M[2,3]

        R = np.reshape(np.concatenate((r_1, r_2, r_3)), (3,3))
        T = np.array([t_x, t_y, t_z])
        intrinsic  = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
        
        extrinsic = np.concatenate((R, np.array([T]).T), axis=1)
        added = np.array([[0,0,0,1]])
        extrinsic_add = np.concatenate((extrinsic, added))
        
        camParameters = {}
        
        camParameters["ImageCoordinate"] = image_coord
        camParameters["WorldCoordinate"] = world_coord
        camParameters["ProjectionMatrix"] = M
        camParameters["Intrinsic"] = intrinsic
        camParameters["Extrinsic"] = extrinsic_add
        camParameters["RotationMatrix"] = R
        camParameters["TranslationMatrix"] = T
        print(camParameters)
        return camParameters
    
        
    def verification(self, image_left,path,  camParameters, intege):
        # image_points_left_temp = self.getPointsFromImage(image_left, "Inputs/image_coordinate.txt", 1)
        
        # image_coord_left = np.array([])
        # for elem in image_points_left_temp:
        #     elem = np.append(elem, [1])
        #     image_coord_left = np.append(image_coord_left, [elem])
        # image_coord_left = np.reshape(image_coord_left, (np.shape(image_points_left_temp)[0],3))
        project  = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        K  = np.matmul(camParameters["Intrinsic"], project)
        world2image = np.matmul(K,camParameters["Extrinsic"])
        
        
        image_coord_left_pred = np.matmul(camParameters["ProjectionMatrix"], camParameters["WorldCoordinate"])
        print("Those are image coordinate")
        print(image_coord_left_pred)
        
        image_coord_left_pred = np.matmul(world2image, camParameters["WorldCoordinate"])
        print("Those are image coordinate")
        print(image_coord_left_pred)
        
        M = np.zeros((3,3))
        for i in range(0,3):
            M[i,0] = world2image[i,0]
            M[i,1] = world2image[i,1]
            M[i,2] = world2image[i,2]

        image2world = np.linalg.pinv(world2image)
        print("image2world matrix", image2world)
        world_coord_left = np.matmul(camParameters["WorldCoordinate"].T, image2world)
        print("Those are world coordinate")
        print(world_coord_left)
        
        ##############################
        tempMat = np.matmul(np.linalg.inv(camParameters["RotationMatrix"]),(np.matmul(np.linalg.inv(camParameters["Intrinsic"]),camParameters["ImageCoordinate"][:,0])))
        tempMat2 = np.matmul(np.linalg.inv(camParameters["RotationMatrix"]),camParameters["TranslationMatrix"])
        s =  tempMat2
        s /= tempMat
        print(s)
        wcPoint = np.matmul(np.linalg.inv(camParameters["RotationMatrix"]),(np.matmul(s,np.matmul(np.linalg.inv(camParameters["Intrinsic"]),camParameters["ImageCoordinate"][:,0])) - camParameters["TranslationMatrix"]))
        ###############################
        if (intege == 0):
            s = 1
            image = cv2.imread("Inputs/left.jpg")
        elif (intege == 1):
            s = -1
            image = cv2.imread("Inputs/right.jpg")
        for elem in camParameters["WorldCoordinate"].T:
            image_coord_test = np.matmul(camParameters["ProjectionMatrix"], elem.T)
            image_coord_test_new = image_coord_test[0:2]
            cv2.circle(image, tuple([s*int(image_coord_test_new[0]), s*int(image_coord_test_new[1])]), 0, color=(0, 0, 255), thickness=5)
        
        if (intege == 0):
            cv2.imwrite("./output/left.jpg", image)
        elif (intege == 1):
            cv2.imwrite("./output/right.jpg", image)

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
            "--right",
            default="Inputs/right.jpg",
            type=str,
            help="Directory to right input images",
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
    
    # LEFT IMAGE
    img_path_left = args.left

    world_coord_file = args.txtfile
    Calibration = Calibration()
    
    image_left = Calibration.PreProcess(img_path_left)
    
    image_points = Calibration.getPointsFromImage(image_left, 'Inputs/image_coordinate.txt', 0)

    M, image_coord, world_coord = Calibration.projectMatrix(image_points, world_coord_file)
    
    camParameters = Calibration.getCameraParameters(M, image_coord, world_coord)
    
    Calibration.verification(image_left,img_path_left,  camParameters, 0)
    
    #RIGHT IMAGE
    
    img_path_right = args.right

    world_coord_file = args.txtfile
    
    image_right = Calibration.PreProcess(img_path_right)
    
    image_points = Calibration.getPointsFromImage(image_right, 'Inputs/image_coordinate_right.txt', 1)

    M, image_coord, world_coord = Calibration.projectMatrix(image_points, world_coord_file)
    
    camParameters = Calibration.getCameraParameters(M, image_coord, world_coord)
    
    Calibration.verification(image_right,img_path_right,  camParameters, 1)