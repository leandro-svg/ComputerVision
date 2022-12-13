import numpy as np
import argparse
import cv2
import cv2 as cv
import math
import time 
import tqdm
import os
from skimage.transform import warp, ProjectiveTransform
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.plot import plot_cube, plot_structure, plot_triangle
from utils.general import click_event, random_with_N_digits, txt2array, txt2array_img
from utils.general import ToHomogeneous
from utils.math import SVD, SVD_coord
# plt.style.use('seaborn-poster')


class Calibration():
    def __init__(self, ):
        super().__init__()
    def PreProcess(self, img_path):
        
        image = cv2.imread(img_path)
        return image

    def getPointsFromImage(self, image3D, path, choice):
        if not(os.path.isfile(path)):
            cv2.imshow('image', image3D)
            cv2.setMouseCallback('image', click_event, (image3D, path))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Image Coordinate file already exists, we will work with it")
        image_coord = txt2array_img(path)
        return image_coord

    def projectMatrix(self, image_coord_temp, world_coord_file):
        world_coord_temp = txt2array(world_coord_file)
        
        world_coord = ToHomogeneous(world_coord_temp, 4)
        image_coord = ToHomogeneous(image_coord_temp, 3)
     
        P = np.array([])
        for i in range(np.shape(image_coord)[0]):
            first = np.concatenate((world_coord[i,:], np.zeros((1,4))[0], -1*image_coord[i,0]*world_coord[i,:]))
            second = np.concatenate((np.zeros((1,4))[0], world_coord[i,:],  -1*image_coord[i,1]*world_coord[i,:]))
            P = np.append(P, first, axis=0)
            P = np.append(P, second, axis=0)
            
        P = np.reshape(P, (24, 12))
        M = SVD(P)
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
        project  = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        K  = np.matmul(camParameters["Intrinsic"], project)
        world2image = np.matmul(K,camParameters["Extrinsic"])
        
        if (intege == 0):
            image = cv2.imread("Inputs/left.jpg")
        elif (intege == 1):
            image = cv2.imread("Inputs/right.jpg")
        for elem in camParameters["WorldCoordinate"].T:
            image_coord_test = np.matmul(world2image, elem.T)
            homogenous = image_coord_test[2]
            image_coord_test_new = image_coord_test[0:2]/homogenous
            cv2.circle(image, tuple([int(image_coord_test_new[0]), int(image_coord_test_new[1])]), 0, color=(0, 0, 255), thickness=5)
        if (intege == 0):
            cv2.imwrite("./output/calibration/reconstructed_left_monocular.jpg", image)
        elif (intege == 1):
            cv2.imwrite("./output/calibration/reconstructed_right_monocular.jpg", image)
     
    def threeDReconstruation(self,image_3D_left,image_3D_right, path_3D_left, path_3D_right, camParameters_right, camParameters_left ):
        image_points_3D_left = (self.getPointsFromImage(image_3D_left, path_3D_left, 2)).T
        image_points_3D_right = (self.getPointsFromImage(image_3D_right, path_3D_right, 3)).T
        
        project_left  = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        K_left  = np.matmul(camParameters_left["Intrinsic"], project_left)
        world2image_left= np.matmul(K_left,camParameters_left["Extrinsic"])
        
        project_right  = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        K_right  = np.matmul(camParameters_right["Intrinsic"], project_right)
        world2image_right = np.matmul(K_right,camParameters_right["Extrinsic"])
        
        M_left = world2image_left
        M_right = world2image_right
        
        m_1_left = M_left[0,:]
        m_2_left = M_left[1,:]
        m_3_left = M_left[2,:]
        m_1_right = M_right[0,:]
        m_2_right =M_right[1,:]
        m_3_right = M_right[2,:]
        
        x_right = image_points_3D_right[0,:]
        y_right = image_points_3D_right[1,:]
        x_left = image_points_3D_left[0,:]
        y_left = image_points_3D_left[1,:]
        
        world_coord = camParameters_left["WorldCoordinate"]
        pred_world_mat = np.zeros((np.shape(image_points_3D_left)[1], np.shape(world_coord)[0]))
        for i in range(np.shape(image_points_3D_left)[1]):
            A = np.array([(x_right[i]*m_3_right)-m_1_right,
                        (y_right[i]*m_3_right)-m_2_right,
                        (x_left[i]*m_3_left)-m_1_left,
                        (y_left[i]*m_3_left)-m_2_left])
            
            pred_world = SVD_coord(A)
            pred_world_mat[i,:] = pred_world

        predicted_world_point = pred_world_mat[:,0:3]
        pannel = predicted_world_point[0:12, :]
        mse = (np.square(world_coord[0:3,:] - pannel.T)).mean()
        print("Mean Squared Error between Real World Coordinates: and Predicted World Coordinates : ", mse) 
        return predicted_world_point, image_points_3D_right, image_points_3D_left
        
    
    def reconstruction(self, predicted_world_point):
        fig = plt.figure(figsize = (7,7))
        ax = plt.axes(projection='3d')
        pannel = predicted_world_point[0:12, :]
        big_cube = predicted_world_point[12:19, :]
        pyramid = predicted_world_point[19:23, :]
        little_cube = predicted_world_point[23:32, :]
        
        polycube = plot_cube(big_cube)
        polycube_little = plot_cube(little_cube)
        polystructure = plot_structure(pannel)
        polytriangle = plot_triangle(pyramid)
        structure = [polycube, polycube_little, polystructure, polytriangle ]
        
        for elem in structure:
            face_colors = '#' + str(random_with_N_digits(6))
            poly = Poly3DCollection(elem,facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.8)
            ax.add_collection3d(poly)
            
        ax.scatter(predicted_world_point[:,0], predicted_world_point[:,1], predicted_world_point[:,2], c = 'r', s = 50)
        plt.savefig("output/calibration/3D_reconstruction.jpg")
        plt.show()
        
    def computeCameraEye_PT(self, image_points, world_points):
        focal_length = 25
        pixel_size = 0.00345
        image_resolution = [2464, 2056]
        image2world = cv2.getPerspectiveTransform(image_points, world_points)

        intrinsic = [[focal_length / pixel_size, 0, image_resolution[0] / 2],
                    [0, focal_length / pixel_size, image_resolution[1] / 2],
                    [0, 0, 1]]

        world_points_3D = []
        image_points_2D = []
        for elem in world_points : 
            world_points_3D.append([elem[0], elem[1], 0])

        for elem in image_points : 
            image_points_2D.append([elem[0], elem[1], 0])
        
        retVal, rvec, tvec = cv2.solvePnP(np.array(world_points_3D), np.array(image_points), np.array(intrinsic), None)
        if retVal :
            rotation_matrix = cv2.Rodrigues(np.array(rvec))[0]
            cameraEye = - np.transpose(rotation_matrix) @ tvec
    
def main(calib, args):
    
    img_path = args.input
    
    # LEFT IMAGE
    img_path_left = args.left
    world_coord_file_left = args.txtfile
    
    image_left = calib.PreProcess(img_path_left)
    image_points_left = calib.getPointsFromImage(image_left, 'Inputs/precomputed_points/LEFTCAL.txt', 0)
    M_left, image_coord_left, world_coord_left = calib.projectMatrix(image_points_left, world_coord_file_left)
    camParameters_left = calib.getCameraParameters(M_left, image_coord_left, world_coord_left)
    calib.verification(image_left,img_path_left,  camParameters_left, 0)
    
    #RIGHT IMAGE
    img_path_right = args.right
    world_coord_file_right = args.txtfile
    image_right = calib.PreProcess(img_path_right)
    image_points_right = calib.getPointsFromImage(image_right, 'Inputs/precomputed_points/RIGHTCAL.txt', 1)
    M_right, image_coord_right, world_coord_right = calib.projectMatrix(image_points_right, world_coord_file_right)
    camParameters_right = calib.getCameraParameters(M_right, image_coord_right, world_coord_right)
    calib.verification(image_right,img_path_right,  camParameters_right, 1)
    
    
    # STEREO CALIBRATION
    image_3D_left = calib.PreProcess(img_path_left)
    image_3D_right = calib.PreProcess(img_path_right)
    predicted_world_point,image_points_3D_right, image_points_3D_left = calib.threeDReconstruation(image_3D_left, image_3D_right, 'Inputs/precomputed_points/LEFTIMG1.txt',
                                                            'Inputs/precomputed_points/RIGHTIMG1.txt', camParameters_right, camParameters_left)
    calib.reconstruction(predicted_world_point)
    
 
        
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
    calib = Calibration()
    args = get_Parser().parse_args()
    main(calib, args)
    
