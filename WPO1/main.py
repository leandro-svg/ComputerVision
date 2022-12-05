import numpy as np
import argparse
import cv2
import math
import time 
import tqdm
import os
from random import randint

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plt.style.use('seaborn-poster')

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def click_event(event, x, y, flags, params):
    print(params)
    image3D, path = params
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image3D, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)

        with open(path, 'a') as f:
            f.write(str(x) + ',' + str(y))
            f.write('\n')
        cv2.imshow('image', image3D)

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
 
        P = np.array([])
        for i in range(np.shape(image_coord)[0]):
            first = np.concatenate((world_coord[i,:], np.zeros((1,4))[0], -1*image_coord[i,0]*world_coord[i,:]))
            second = np.concatenate((np.zeros((1,4))[0], world_coord[i,:],  -1*image_coord[i,1]*world_coord[i,:]))
            P = np.append(P, first, axis=0)
            P = np.append(P, second, axis=0)
            
        P = np.reshape(P, (24, 12))
        u, s, vh = np.linalg.svd(P)
        v = vh.T
        Mbis = v[:, -1].T
        Mbis = np.reshape(Mbis, (3,4))
        norm_3 = np.linalg.norm(Mbis[2,:])
        Mbis = -1*Mbis/norm_3
        return Mbis, image_coord.T, world_coord.T

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
            s = 1
            image = cv2.imread("Inputs/left.jpg")
        elif (intege == 1):
            s = 1
            image = cv2.imread("Inputs/right.jpg")
        for elem in camParameters["WorldCoordinate"].T:
            image_coord_test = np.matmul(world2image, elem.T)
            homogenous = image_coord_test[2]
            image_coord_test_new = image_coord_test[0:2]/homogenous
            cv2.circle(image, tuple([s*int(image_coord_test_new[0]), s*int(image_coord_test_new[1])]), 0, color=(0, 0, 255), thickness=5)
        if (intege == 0):
            cv2.imwrite("./output/reconstructed_left_monocular.jpg", image)
        elif (intege == 1):
            cv2.imwrite("./output/reconstructed_right_monocular.jpg", image)
     
    def threeDReconstruation(self,image_3D_left,image_3D_right, path_3D_left, path_3D_right, camParameters_right, camParameters_left ):
        image_points_3D_left = (Calibration.getPointsFromImage(image_3D_left, path_3D_left, 2)).T
        image_points_3D_right = (Calibration.getPointsFromImage(image_3D_right, path_3D_right, 3)).T
        
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
            
            u, s, vh = np.linalg.svd(A)
            v = vh.T
            pred_world = v[:, -1]
            norm_3 = (pred_world[-1])
            pred_world = (pred_world/norm_3)
            pred_world_mat[i,:] = pred_world

        predicted_world_point = pred_world_mat[:,0:3]
        pannel = predicted_world_point[0:12, :]
        mse = (np.square(world_coord[0:3,:] - pannel.T)).mean()
        print("Mean Squared Error between Real World Coordinates: and Predicted World Coordinates : ", mse) 
        return predicted_world_point
        
    def plot_cube(self, structure):
        Z = np.zeros([8, 3])
        Z[0:7, :] = np.array(structure[0:7])
        Z[7, :] = np.array([Z[6, 0], Z[4, 1], Z[5, 2]])
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                [Z[4], Z[5], Z[6], Z[7]],
                [Z[0], Z[1], Z[7], Z[4]],
                [Z[2], Z[3], Z[5], Z[6]],
                [Z[1], Z[2], Z[6], Z[7]],
                [Z[4], Z[7], Z[1], Z[0]]]
        return verts
    
    def plot_structure(self, pannel):
        a = pannel
        Z = np.zeros([6, 3])
        Z[0, :] = np.array([0, 0, 0])
        Z[1, :] = np.array([0, 0, pannel[11, 2]])
        Z[2, :] = np.array(pannel[10])
        Z[3, :] = np.array([0, pannel[6][1], 0])
        Z[4, :] = np.array(pannel[5])
        Z[5, :] = np.array([pannel[1][0], 0, 0])
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                [Z[0], Z[1], Z[4], Z[5]]]
        return verts
    
    def plot_triangle(self, structure):
        Z = np.zeros([4, 3])
        Z[0:4, :] = np.array(structure[0:4])
        verts = [[Z[0], Z[1], Z[2]],
                [Z[0], Z[1], Z[3]],
                [Z[3], Z[0], Z[2]],
                [Z[0], Z[2], Z[3]]]
        return verts
    
    def reconstruction(self, predicted_world_point):
        fig = plt.figure(figsize = (7,7))
        ax = plt.axes(projection='3d')
        pannel = predicted_world_point[0:12, :]
        big_cube = predicted_world_point[12:19, :]
        pyramid = predicted_world_point[19:23, :]
        little_cube = predicted_world_point[23:32, :]
        
        polycube = self.plot_cube(big_cube)
        polycube_little = self.plot_cube(little_cube)
        polystructure = self.plot_structure(pannel)
        polytriangle = self.plot_triangle(pyramid)
        structure = [polycube, polycube_little, polystructure, polytriangle ]
        
        for elem in structure:
            face_colors = '#' + str(random_with_N_digits(6))
            poly = Poly3DCollection(elem,facecolors=face_colors, linewidths=1, edgecolors='k', alpha=0.8)
            ax.add_collection3d(poly)
            
        ax.scatter(predicted_world_point[:,0], predicted_world_point[:,1], predicted_world_point[:,2], c = 'r', s = 50)
        plt.savefig("output/3D_reconstruction.jpg")
        plt.show()
        
        
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
    world_coord_file_left = args.txtfile
    Calibration = Calibration()
    image_left = Calibration.PreProcess(img_path_left)
    image_points_left = Calibration.getPointsFromImage(image_left, 'Inputs/image_coordinate.txt', 0)
    M_left, image_coord_left, world_coord_left = Calibration.projectMatrix(image_points_left, world_coord_file_left)
    camParameters_left = Calibration.getCameraParameters(M_left, image_coord_left, world_coord_left)
    Calibration.verification(image_left,img_path_left,  camParameters_left, 0)
    
    
    img_path_right = args.right
    world_coord_file_right = args.txtfile
    image_right = Calibration.PreProcess(img_path_right)
    image_points_right = Calibration.getPointsFromImage(image_right, 'Inputs/image_coordinate_right.txt', 1)
    M_right, image_coord_right, world_coord_right = Calibration.projectMatrix(image_points_right, world_coord_file_right)
    camParameters_right = Calibration.getCameraParameters(M_right, image_coord_right, world_coord_right)
    Calibration.verification(image_right,img_path_right,  camParameters_right, 1)
    
    
    # STEREO CALIBRATION
    image_3D_left = Calibration.PreProcess(img_path_left)
    image_3D_right = Calibration.PreProcess(img_path_right)
    predicted_world_point = Calibration.threeDReconstruation(image_3D_left, image_3D_right, 'Inputs/3D_image_coordinate_left.txt', 'Inputs/3D_image_coordinate_right.txt', camParameters_right, camParameters_left)
    Calibration.reconstruction(predicted_world_point)