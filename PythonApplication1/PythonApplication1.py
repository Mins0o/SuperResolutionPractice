print("pip install numpy opencv matplotlib")
print("Importing openCV                 ",end = "\r")
import matplotlib.pyplot as plt
print("Importing numpy                  ",end = "\r")
import numpy as np
print("Importing openCV                 ",end = "\r")
import cv2
import os



""" download models from the urls below
https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/EDSR_x2.pb?raw=true
https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/EDSR_x3.pb?raw=true
https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/EDSR_x4.pb?raw=true

https://github.com/fannymonori/TF-ESPCN/blob/master/export/ESPCN_x2.pb?raw=true
https://github.com/fannymonori/TF-ESPCN/blob/master/export/ESPCN_x3.pb?raw=true
https://github.com/fannymonori/TF-ESPCN/blob/master/export/ESPCN_x4.pb?raw=true

https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN-small_x2.pb?raw=true
https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN-small_x2.pb?raw=true
https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN-small_x2.pb?raw=true

https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN_x2.pb?raw=true
https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN_x3.pb?raw=true
https://github.com/Saafke/FSRCNN_Tensorflow/blob/master/models/FSRCNN_x4.pb?raw=true

https://github.com/fannymonori/TF-LapSRN/blob/master/export/LapSRN_x2.pb?raw=true
https://github.com/fannymonori/TF-LapSRN/blob/master/export/LapSRN_x4.pb?raw=true
https://github.com/fannymonori/TF-LapSRN/blob/master/export/LapSRN_x8.pb?raw=true
"""

image_extensions = [".jpg",".png"]
base_path = "./"

def load_inputs(base_path= "./", image_dir = "input_images"):
    image_dir = base_path + image_dir
    all_names = os.listdir(image_dir)
    image_names = [file_name for file_name in all_names if file_name[-4:] in image_extensions]
    images = [cv2.imread(image_dir +"/"+ img_name) for img_name in image_names]
    return images

def show_image_grid(images):
    num_images = len(images)
    biggest_factor = 1
    biggest_addition = 0

    # determine grid height and width with the most suitable factors
    for filler in range(int(num_images/6)+1):
        factoring_number = num_images + filler
        factor_mid_point = int(np.sqrt(factoring_number))+1
        for factor_try in range(1,factor_mid_point):
            if (factoring_number) % factor_try == 0 and factor_try > biggest_factor:
                biggest_factor = factor_try
                biggest_addition = filler
    if (num_images+biggest_addition)/biggest_factor/biggest_factor <3:
        grid_height = biggest_factor
        grid_width = (num_images+biggest_addition)/biggest_factor
    else:
        grid_height = np.round(np.sqrt(num_images))
        grid_width = np.ceil(np.sqrt(num_images))

    # create new window and display images in subplots
    plt.figure()
    for img in range(num_images):
        plt.subplot(int(grid_height), int(grid_width), img+1)
        plt.axis("off")
        plt.margins(0)
        plt.imshow(images[img][:,:,::-1])
    plt.show()
            


images = load_inputs()
#show_image_grid(images)

width_crop = lambda img,start,end: slice(int(img.shape[0]*start), int(img.shape[0]*end))
height_crop = lambda img,start,end: slice(int(img.shape[1]*start), int(img.shape[1]*end))

cropped =  [img[width_crop(img,1/3,1/2), height_crop(img,1/3,1/2)] for img in images]
show_image_grid(cropped)