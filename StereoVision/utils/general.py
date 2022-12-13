import cv2
import numpy as np
from random import randint

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
        
def txt2array(world_coord_file):  
    with open(world_coord_file, 'r') as file:
        world_lines = [line.rstrip() for line in file]
    while("" in world_lines):
        world_lines.remove("")     
    world_coord_temp = []
    for elem in world_lines: 
        for coord in elem.split('   ') :
            world_coord_temp.append(int(coord))
    world_coord_temp = np.reshape(world_coord_temp, (np.size(world_lines),3))
    return world_coord_temp

def txt2array_img(path):
    with open(path, 'r') as file:
        lines = [line.rstrip() for line in file]

    image_coord = []
    for elem in lines: 
        for coord in elem.split(',') : 
            image_coord.append(int(coord))
    image_coord = np.reshape(image_coord, (np.size(lines),2))
    return image_coord

def ToHomogeneous(world_coord_temp, shape):
    world_coord = np.array([])
    for elem in world_coord_temp:
        elem = np.append(elem, [1])
        world_coord = np.append(world_coord, [elem])
    world_coord = np.reshape(world_coord, (np.shape(world_coord_temp)[0],shape))
    return world_coord

def write_file(camParameters):
    file_path = 'Parameters.txt'
  
    with open(file_path, 'a') as f:
        for key, value in camParameters.items(): 
            f.write('%s:%s\n' % (key, value))
        # convert_file.write(str(camParameters))
    with open(file_path, 'a') as convert_file:
        convert_file.write(str('\n\n'))