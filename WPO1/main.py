import numpy as np
import argparse
import cv2
import cv2 as cv
import math
import time 
import tqdm
import os
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
        return predicted_world_point, image_points_3D_right, image_points_3D_left
        
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
    
    def check(self):
        print("We good in here")
        
        
class Epipolar(Calibration):
    def __init__(self, ):
        super().__init__()
        #Sources:
        #https://towardsdatascience.com/a-comprehensive-tutorial-on-stereo-geometry-and-stereo-rectification-with-python-7f368b09924a
        #https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html
        #https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga49ea1a98c80a5e7d50ad4361dcf2877a
        
    def epiParameters(self, image_3D_left,image_3D_right, camParameters_right, camParameters_left, image_points_3D_right, image_points_3D_left):
        project_left  = np.array([[1,0,0],[0,1,0],[0,0,1]])
        K_left  = np.matmul(camParameters_left["Intrinsic"], project_left)
        
        project_right  = np.array([[1,0,0],[0,1,0],[0,0,1]])
        K_right  = np.matmul(camParameters_right["Intrinsic"], project_right) 
     
        
        ptsRight = image_points_3D_right[:,:].T
        ptsLeft = image_points_3D_left[:,:].T
        
        FundMat, mask = cv2.findFundamentalMat(ptsLeft,ptsRight,cv2.FM_LMEDS)
        EssMat = np.matmul(K_right.T, np.matmul(FundMat, K_left))
        
        new_ptsLeft = np.append(ptsLeft[0,:], [1])
        new_ptsRight = np.append(ptsRight[0,:], [1])
        p_left = np.matmul(np.linalg.pinv(K_left), new_ptsLeft)
        p_right = np.matmul(np.linalg.pinv(K_right), new_ptsRight)
        
        u, s, vh = np.linalg.svd(FundMat)
        v = vh.T
        e = v[:, -1]
        norm_3 = (e[-1])
        e_left = (e/norm_3)
        
        u, s, vh = np.linalg.svd(FundMat.T)
        v = vh.T
        e_2 = v[:, -1]
        norm_3 = e_2[-1]
        e_right = (e_2/norm_3)
        
        print("This should be zero : ",np.round(e_right.T @ FundMat @ e_left))

        img1 = cv.imread("Inputs/left.jpg")
        color = tuple(np.random.randint(0,255,3).tolist())
        for elem in ptsLeft:
            img1 = cv.line(img1, (elem[0],elem[1]), (int(e_left[0]),int(e_left[1])), color,1)
            img1 = cv.circle(img1,(elem[0],elem[1]),5,color,-1)
        img2 = cv.imread("Inputs/right.jpg")
        for elem in ptsRight:
            img2 = cv.line(img2, (elem[0],elem[1]), (int(e_right[0]),int(e_right[1])), color,1)
            img2 = cv.circle(img2,(elem[0],elem[1]),5,color,-1)
        cv.imwrite("output/epipolar/epi_lines_left.jpg", img1)
        cv.imwrite("output/epipolar/epi_lines_right.jpg", img2)
        
        
        return e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat
    
    def compute_matching_homographies(self, e2, F, im2, points1_, points2_):
  
        points1 = np.array([])
        for elem in points1_:
            elem = np.append(elem, [1])
            points1 = np.append(points1, [elem])
        points1 = np.reshape(points1, (np.shape(points1_)[0],3))
        
        points2 = np.array([])
        for elem in points2_:
            elem = np.append(elem, [1])
            points2 = np.append(points2, [elem])
        points2 = np.reshape(points2, (np.shape(points2_)[0],3))
        
        
        p1 = points1.T[:, 0]
        p2 = points2.T[:, 0]
        print("This should be zero : ",np.round(p2.T @ FundMat @ p1))
        
        h, w, rgb = im2.shape
        T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
        
        e2_p = T @ e2
        e2_p = e2_p / e2_p[2]
        e2x = e2_p[0]
        e2y = e2_p[1]
        # create the rotation matrix to rotate the epipole back to X axis
        if e2x >= 0:
            a = 1
        else:
            a = -1
        R1 = a * e2x / np.sqrt(e2x ** 2 + e2y ** 2)
        R2 = a * e2y / np.sqrt(e2x ** 2 + e2y ** 2)
        R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
        e2_p = R @ e2_p
        x = e2_p[0]
        G = np.array([[1, 0, 0], [0, 1, 0], [-1/x, 0, 1]])
        H2 = np.linalg.inv(T) @ G @ R @ T

        e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
        M = e_x @ F + e2.reshape(3,1) @ np.array([[1, 1, 1]])
        points1_t = H2 @ M @ points1.T
        points2_t = H2 @ points2.T
        points1_t /= points1_t[2, :]
        points2_t /= points2_t[2, :]
        b = points2_t[0, :]
        a = np.linalg.lstsq(points1_t.T, b, rcond=None)[0]
        H_A = np.array([a, [0, 1, 0], [0, 0, 1]])
        H1 = H_A @ H2 @ M
        return H1, H2, points1, points2

    def rectification(self, im1, im2, e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat):
        H1, H2, points1, points2 = self.compute_matching_homographies(e_right, FundMat, im1, ptsLeft,ptsRight )
 
        h, w, rgb = im2.shape
        im1_warped = cv2.warpPerspective(im1, H2, (w,h))
        im2_warped = cv2.warpPerspective(im2, H1, (w,h))

        cv.imwrite("output/epipolar/wraped_right.jpg", im1_warped)
        cv.imwrite("output/epipolar/wraped_left.jpg", im2_warped)
        
        new_points1 = H1 @ points1.T
        new_points2 = H2 @ points2.T
        new_points1 /= new_points1[2,:]
        new_points2 /= new_points2[2,:]
        new_points1 = new_points1.T
        new_points2 = new_points2.T
        
        
        
        img1 = cv.imread("output/epipolar/wraped_right.jpg")
        color = tuple(np.random.randint(0,255,3).tolist())
        for elem in new_points2:
            img1 = cv.line(img1, (int(elem[0]),int(elem[1])), (int(e_right[0]),int(e_right[1])), color,1)
            img1 = cv.circle(img1,(int(elem[0]),int(elem[1])),5,color,-1)
        img2 = cv.imread("output/epipolar/wraped_left.jpg")
        for elem in new_points1:
            img2 = cv.line(img2, (int(elem[0]),int(elem[1])), (int( e_left[0]),int(e_left[1])), color,1)
            img2 = cv.circle(img2,(int(elem[0]),int(elem[1])),5,color,-1)
        cv.imwrite("output/epipolar/epi_lines_right_reconstruct.jpg", img1)
        cv.imwrite("output/epipolar/epi_lines_left_reconstruct.jpg", img2)


        im1_warped = cv.imread("output/epipolar/wraped_right.jpg")
        im2_warped = cv.imread("output/epipolar/wraped_left.jpg")

        

        nrows = 2
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))

        # plot image 1
        ax1 = axes[0]
        ax1.set_title("Image 1 warped")
        ax1.imshow( im2_warped, cmap="gray")

        # plot image 2
        ax2 = axes[1]
        ax2.set_title("Image 2 warped")
        ax2.imshow(im1_warped, cmap="gray")

        # plot the epipolar lines and points
        n = new_points1.shape[0]
        for i in range(n):
            p1 = new_points1[i]
            p2 = new_points2[i]

            ax1.hlines(p2[1], 0, w, color="orange")
            ax1.scatter(*p1[:2], color="blue")

            ax2.hlines(p1[1], 0, w, color="orange")
            ax2.scatter(*p2[:2], color="blue")
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
    image_points_left = Calibration.getPointsFromImage(image_left, 'Inputs/precomputed_points/image_coordinate_left.txt', 0)
    M_left, image_coord_left, world_coord_left = Calibration.projectMatrix(image_points_left, world_coord_file_left)
    camParameters_left = Calibration.getCameraParameters(M_left, image_coord_left, world_coord_left)
    Calibration.verification(image_left,img_path_left,  camParameters_left, 0)
    
    #RIGHT IMAGE
    img_path_right = args.right
    world_coord_file_right = args.txtfile
    image_right = Calibration.PreProcess(img_path_right)
    image_points_right = Calibration.getPointsFromImage(image_right, 'Inputs/precomputed_points/image_coordinate_right.txt', 1)
    M_right, image_coord_right, world_coord_right = Calibration.projectMatrix(image_points_right, world_coord_file_right)
    camParameters_right = Calibration.getCameraParameters(M_right, image_coord_right, world_coord_right)
    Calibration.verification(image_right,img_path_right,  camParameters_right, 1)
    
    
    # STEREO CALIBRATION
    image_3D_left = Calibration.PreProcess(img_path_left)
    image_3D_right = Calibration.PreProcess(img_path_right)
    predicted_world_point,image_points_3D_right, image_points_3D_left = Calibration.threeDReconstruation(image_3D_left, image_3D_right, 'Inputs/precomputed_points/3D_image_coordinate_left.txt',
                                                             'Inputs/precomputed_points/3D_image_coordinate_right.txt', camParameters_right, camParameters_left)
    Calibration.reconstruction(predicted_world_point)
    
    #EPIPOLAR LINES
    epi = Epipolar()
    e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat = epi.epiParameters(image_3D_left, image_3D_right, camParameters_right, camParameters_left, image_points_3D_right, image_points_3D_left)
    epi.rectification(image_3D_right, image_3D_left, e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat)