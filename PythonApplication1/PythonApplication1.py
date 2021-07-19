print("pip install numpy opencv matplotlib")
print("> Importing openCV                 ",end = "\r")
import matplotlib.pyplot as plt
print("> Importing numpy                  ",end = "\r")
import numpy as np
print("> Importing openCV                 ",end = "\r")
import cv2
import os
print("> Executing the code               ",)
import matplotlib


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

class UpscaleTester:
    def __init__(self, base_dir = "./"):
        self.base_dir = base_dir
        self.__model__ = cv2.dnn_superres.DnnSuperResImpl_create()
   
    def __grid_dimensions__(self, num_elem:int, target_ratio:float = (16/9.0)) -> (int, int):
        fittest_factor = 1
        fittest_score = 99999
        fittest_addition = 0

        # determine grid height and width with the most suitable factors
        for filler in range(int(num_elem/6)+1):
            factoring_number = num_elem + filler
            factor_mid_point = int(np.sqrt(factoring_number))+1
            for factor_try in range(1,factor_mid_point):
                fit_score = np.abs((num_elem+filler)/(factor_try**2) - target_ratio)
                if (factoring_number) % factor_try == 0 and fit_score < fittest_score:
                    fittest_factor = factor_try
                    fittest_addition = filler
                    fittest_score = fit_score
        grid_height = fittest_factor
        grid_width = (num_elem+fittest_addition)/fittest_factor
        return(grid_height, grid_width)

    def crop_img_list(self, img_list, x_start = 1/2, x_end = 3/5, y_start = 1/2, y_end = 3/5):
        width_crop = lambda img,start,end: slice(int(img.shape[0]*start), int(img.shape[0]*end))
        height_crop = lambda img,start,end: slice(int(img.shape[1]*start), int(img.shape[1]*end))
        return [img[width_crop(img, x_start, x_end), height_crop(img, y_start, y_end)] for img in img_list]
 
    def read_set_model(self, model, scale) -> None:
        print("> Searching in \n{0}models/ \nfor {1}_x{2}.pb model".format(self.base_dir, model.upper(), scale))
        model_path = "{0}models/{1}_x{2}.pb".format(self.base_dir, model.upper(), scale)
        if os.path.isfile(model_path):
            self.__model__.readModel(model_path)
            self.__model__.setModel(model.lower(), scale)
            print("> Set the model to {0}_x{1}".format(model.upper(),scale))
        else:
            print("> The model file \n{}\ndoesn't exist".format(model_path))

    def get_model_n_scale(self) -> (str, int):
        model_name = self.__model__.getAlgorithm()
        scale_value = self.__model__.getScale()
        return(model_name, scale_value)

    def upscale_repeat(self, img, scale:int) -> np.ndarray:
        original_dtype = img.dtype
        o = np.ones((scale,scale))
        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]
        return np.array(np.stack((np.kron(ch1,o), np.kron(ch2,o), np.kron(ch3,o)),axis = 2), dtype = original_dtype)

    def upscale_dnn(self, img:np.ndarray):
        return self.__model__.upsample(img)

    def upscale_repeat_list(self, img_list, scale = 4) -> list:
        return [self.upscale_repeat(img,scale) for img in img_list]

    def upscale_dnn_list(self, img_list, model:str = None, scale:int = 4, verbose:bool = False):
        if not model == None:
            old = self.get_model_n_scale()
            try:
                self.read_set_model(model, scale)
            except e:
                print('The model "{0}_{1}.pb" couldn`t be loaded'.format(model.upper(),scale))
                print(e)
                return img_list
        print("Upscaling {} images".format(len(img_list)))
        if verbose:
            results = []
            for img_num in range(len(img_list)):
                print("image {:3d}/{:3d} being processed       ".format(img_num+1,len(img_list)),end= "\r")
                results.append(self.__model__.upsample(img_list[img_num]))
            print()
        else:
            results = [self.__model__.upsample(img) for img in img_list]
        if not model == None:
            read_set_model = (old[0],old[1])
        print("-----------------------------------------")
        return results
    
    def fig_image_grid(self, img_list) -> matplotlib.figure.Figure:
        num_images = len(img_list)
        grid_height, grid_width = self.__grid_dimensions__(num_images)

        # create new window and display images in subplots
        fig = plt.figure()
        for img in range(num_images):
            plt.subplot(int(grid_height), int(grid_width), img+1)
            plt.axis("off")
            plt.margins(0)
            plt.imshow(img_list[img][:,:,::-1])
        return(fig)
          
    def fig_comp_grid(self, img_list_1, img_list_2, axis = 0) -> matplotlib.figure.Figure:
        """
        The two lists' elements should match each other, and the scale difference should be consistent.
        axis = 0 : vertical concatenation
        axis = 1 : horizontal concatenation
        """
        
        num_elem = len(img_list_1)
        if not len(img_list_2) == num_elem:
            raise Exception("Number of images doesn't match for the two lists\nimg_list_1: {0}  img_list_2: {1}".format(num_elem, len(img_list_2)))
        lists = [img_list_1, img_list_2]
        dims = (len(img_list_1[0]), len(img_list_2[0]))

        small = 0
        big = 1
        if dims[0] > dims[1]:
            small = 1
            big = 0

        if dims[big]%dims[small]:
            raise Exception("The scale is not integer. \nimg_list_1: {0} \nimg_list_2: {1}".format(len(img_list_1),len(img_list_2)))

        scale = int(dims[big]/dims[small])
        lists[small] = self.upscale_repeat_list(lists[small], scale)
        comp = [np.concatenate((lists[0][i], lists[1][i]),axis = axis) for i in range(num_elem)]
        return self.fig_image_grid(comp)

    def fig_comp_dnn_orig(self, img_list, axis = 1, model:str = None, scale:int = 4, verbose = False):
        upscaled = self.upscale_dnn_list(img_list, model, scale, verbose)
        return (self.fig_comp_grid(img_list, upscaled, axis))

def load_inputs(base_path= "./", image_dir = "input_images"):
        image_dir = base_path + image_dir
        all_names = os.listdir(image_dir)
        image_names = [file_name for file_name in all_names if file_name[-4:] in image_extensions]
        images = [cv2.imread(image_dir +"/"+ img_name) for img_name in image_names]
        return images
     

ut = UpscaleTester()

images = load_inputs()
cropped = ut.crop_img_list(images, 1/2, 3/5, 3/5, 5/7)

# ut.fig_image_grid(images)
# ut.fig_image_grid(cropped)

ut.read_set_model("ESPCN",4)
ut.fig_comp_dnn_orig(cropped[:7], axis = 0)
comparison = []
upscaled = []
target = images[7:27]
ut.fig_comp_dnn_orig(target, model = "EDSR", scale = 4, verbose = True)
plt.show()
