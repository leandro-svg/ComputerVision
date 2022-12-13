# Computer Vision : Calibration and Stereo Vision

**This repository implements a manual calibration and a reconstruction of a 3D scene thanks to a stereo vision model. You can also find a rectification of the images with an implementation of parallel epipolar lines.**

 ## Installation and dependencies
 -  Ubuntu 20.04.5 LTS

- ```
  pip3 install requirements.txt
  ```


## Preliminaries 

Precomputed points from images are stored in the inputs/precomputed_points directory. If you want to select those points by yourself, remove those files and run the command in the following section.

 ## Running Calibration 
To run the calibration code, you can enter the following command in your terminal : 
```
python3 main.py --input structure_image_path --left left_image_path --right right_image_path --txtfile world_coordinate_file --parametersFile name_output_txtfile
```
If you haven't change the ZIP file, you can run directly the following command :

```
python3 main.py --input Inputs/calibration_pointnumbering.jpg --left Inputs/left.jpg --right Inputs/right.jpg --txtfile Inputs/calibration_points3.txt --parametersFile Parameters.txt
```

or : ```
      python3 main.py 
      ```

## Running Epipolar : 
To run the epipolar code, you can enter the following command in your terminal : 
```
python3 epipolar.py --input structure_image_path --left left_image_path --right right_image_path --txtfile world_coordinate_file --parametersFile name_output_txtfile
```
If you haven't change the ZIP file, you can run directly the following command :

```
python3 epipolar.py --input Inputs/calibration_pointnumbering.jpg --left Inputs/left.jpg --right Inputs/right.jpg --txtfile Inputs/calibration_points3.txt --parametersFile Parameters.txt
```

or : ```
     python3 epipolar.py 
     ```


## Expected output : 
After having run the two codes (First Calibration, then Epipolar), you can find the following output :
- Parameters.txt : The two parameters for right and left images and the MSE at the end
- In the output/calibration directory : 
    - reconstructed_left_monocular and reconstructed_right_monocular are visualization of the world axes in both left and right images
    - 3D_reconstruction_i are the reconstruction of the scene in 3D for different angles
- In the output/epipolar directory : 
    - epi_lines_left and epi_lines_right are original epipolar lines 
    - epi_lines_left_rectification  and epi_lines_left_rectification  are the images rectified with the parallel lines
