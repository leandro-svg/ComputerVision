# Computer Vision : Calibration and Stereo Vision

**This repository implements a manual calibration and a reconstruction of a 3D scene thanks to a stereo vision model. You can also find a rectification of the images with an implementation of parallel epipolar lines.**

 ## Installation and dependencies

```
pip3 install requirements.txt
```

 ## Running Calibration 
```
python3 main.py --input structure_image_path --left left_image_path --right right_image_path --txtfile world_coordinate_file --parametersFile name_output_txtfile
```
If you haven't change the ZIP file, you can run directly the following command :

 ```
python3 main.py --input Inputs/calibration_pointnumbering.jpg --left Inputs/left.jpg --right Inputs/right.jpg --txtfile Inputs/calibration_points3.txt --parametersFile Parameters.txt
```

or :
```
python3 main.py 
```

