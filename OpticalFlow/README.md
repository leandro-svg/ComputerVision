# Computer Vision : Calibration and Stereo Vision

**This repository (https://github.com/leandro-svg/ComputerVision/tree/master/OpticalFlow) implements a manual implementation of Optical Flow using  : Lucas Kanada and Horn-Chunkz methods.**

 ## Installation and dependencies
 -  Ubuntu 20.04.5 LTS

- ```
  pip3 install -r requirements.txt
  ```


 ## Running Optical Flow Computation 
To run the Optical Flow code, you can enter the following command in your terminal : 
```
python3 main.py --input images_directory_path/* 
```
If you haven't change the directory path, you can run directly the following command :

```
python3 main.py --input Basketball/*
```



## Expected output : 
After having run the code, you can find the following output :
- In the output directory : 
    - Horn_Shunck_OF.png is the optical flow computed for each images returned by Horn-Shunck method
    - lucas_kanade_OF.png is the optical flow computed for each images returned by Lucas_Kanade method


## To go further : 
Additionnaly, to go a bit deeper in Optical Flow computing, one could use Deep Learning method to have more advanced and precise results than manual computation. 
In order to do it, we are going to test one of the well-known architecture for optical flow : RAFT. (Recurrent All-Pairs Field Transforms for
Optical Flow). The result can be found in the report as well.

For installation, run the following commands (On ubuntu) : 
```
git clone https://github.com/princeton-vl/RAFT.git
cd RAFT
./download_models.sh 

```
If the bash script doesn't work, download raft-small/things.pth at https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT .

Additionnaly, you could install the following dependencies if not already done : (Personnally used Pip instead of Conda)
```
conda create --name raft
conda activate raft
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```


To run the model, you can : 
```
python3 demo.py --model=models/raft-things.pth --path=Basketball
```
where the path Basketball is the path of the images used for the initial project.