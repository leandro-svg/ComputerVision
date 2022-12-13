import numpy as np
import argparse
import cv2
import cv2 as cv
from skimage.transform import warp, ProjectiveTransform
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from utils.math import SVD_coord
from main import Calibration, main as main_calib

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
        
        e_left = SVD_coord(FundMat)
        e_right = SVD_coord(FundMat.T)
        
        # print("This should be zero : ", np.round(e_right.T @ FundMat @ e_left))

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
        # print("This should be zero : ",np.round(p2.T @ F @ p1))
        
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
        cv.imwrite("output/epipolar/epi_lines_right_rectification.jpg", img1)
        cv.imwrite("output/epipolar/epi_lines_left_rectification.jpg", img2)

        im1_warped = cv.imread("output/epipolar/wraped_right.jpg")
        im2_warped = cv.imread("output/epipolar/wraped_left.jpg")

        nrows = 2
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 8))

        ax1 = axes[0]
        ax1.set_title("Image 1 warped")
        ax1.imshow( im2_warped, cmap="gray")

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
        plt.savefig("output/epipolar/parallel_warped_reconstructed.jpg")

def main_epi(epi, args):
    #GET PARAM FROM CALIB
    predicted_world_point, image_3D_left, image_3D_right, camParameters_right, camParameters_left, image_points_3D_right, image_points_3D_left = main_calib(args)
    #EPIPOLAR LINES
    e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat = epi.epiParameters(image_3D_left, image_3D_right, camParameters_right, camParameters_left, image_points_3D_right, image_points_3D_left)
    epi.rectification(image_3D_right, image_3D_left, e_right, e_left, ptsRight, ptsLeft, FundMat, EssMat)    

        
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
    parser.add_argument(
            "--parametersFile",
            default="Parameters.txt",
            type=str,
            help="Cam parameters Results",
            )
    return parser


if __name__ == '__main__':
    args = get_Parser().parse_args()
    epi = Epipolar()
    main_epi(epi, args)
    