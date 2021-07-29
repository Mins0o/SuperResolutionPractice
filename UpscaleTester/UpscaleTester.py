print("pip install numpy opencv matplotlib")
print("> Importing openCV                 ",end = "\r")
import matplotlib.pyplot as plt
print("> Importing numpy                  ",end = "\r")
import numpy as np
print("> Importing openCV                 ",end = "\r")
import cv2
import os
print("                                   ")
import matplotlib

class UpscaleTester:
    def __init__(self, base_dir = "./"):
        self.base_dir = base_dir
        self._model = cv2.dnn_superres.DnnSuperResImpl_create()
        self._rename_models()
        """
        edsr | espcn | fsrcnn | fsrcnn-small | lapsrn
        """

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
   
    def _rename_models(self):
        if not os.path.exists(self.base_dir+"/models"):
            os.mkdir(self.base_dir+"/models")
        for file_name in os.listdir(self.base_dir+"/models"):
            if file_name[-3:].lower() == ".pb":
                os.rename(self.base_dir+"/models/"+file_name, self.base_dir+"/models/"+file_name[:-6].upper()+file_name[-6:])
        
    def _grid_dimensions(self, num_elem:int, target_ratio:float = (16/9.0)) -> (int, int):
        fittest_factor = 1
        fittest_score = 99999
        fittest_addition = 0

        # determine grid height and width with the most suitable factors
        for filler in range(int(num_elem/6)+1):
            factoring_number = num_elem + filler
            factors = [i for i in range(1, int(np.sqrt(factoring_number))+1) if not factoring_number%i]
            factors += [factoring_number//i for i in factors]
            for factor_try in factors:
                fit_score = np.abs((num_elem+filler)/(factor_try**2) - target_ratio)
                if (factoring_number) % factor_try == 0 and fit_score < fittest_score:
                    fittest_factor = factor_try
                    fittest_addition = filler
                    fittest_score = fit_score
        grid_height = fittest_factor
        grid_width = (num_elem+fittest_addition)/fittest_factor
        return(grid_height, grid_width)

    def crop_img_list(self, img_list, x_start = 1/2, x_end = 3/5, y_start = 1/2, y_end = 3/5, height = None, width = None):
        start_end = lambda img, axis, start, end: slice(int(img.shape[axis]*start), int(img.shape[axis]*end))
        start_size = lambda img, axis, start, size: slice(int(img.shape[axis]*start), int(img.shape[axis]*start) + size)

        if height == None:
            x_crop = lambda img: start_end(img, 0, x_start, x_end)
        else: 
            x_crop = lambda img: start_size(img, 0, x_start, height)
        if width == None:
            y_crop = lambda img: start_end(img, 1, y_start, y_end)
        else:
            y_crop = lambda img: start_size(img, 1, y_start, width)
        return [img[x_crop(img), y_crop(img)] for img in img_list]
 
    def read_set_model(self, model, scale, verbose = False) -> None:
        if verbose:
            print("> Searching in \n{0}models/ \nfor {1}_x{2}.pb model".format(self.base_dir, model.upper(), scale))
        model_path = "{0}models/{1}_x{2}.pb".format(self.base_dir, model.upper(), scale)
        if os.path.isfile(model_path):
            self._model.readModel(model_path)
            self._model.setModel(model.lower().split("-")[0], scale)
            if verbose:
                print("> Set the model to {0}_x{1}".format(model.upper(),scale))
        else:
            print("> The model file \n{}\ndoesn't exist".format(model_path))

    def get_model_n_scale(self) -> (str, int):
        model_name = self._model.getAlgorithm().lower()
        scale_value = self._model.getScale()
        return(model_name, scale_value)

    def upscale_nearest_neighbor(self, img, scale:int) -> np.ndarray:
        original_dtype = img.dtype
        o = np.ones((scale,scale))
        ch1 = img[:,:,0]
        ch2 = img[:,:,1]
        ch3 = img[:,:,2]
        return np.array(np.stack((np.kron(ch1,o), np.kron(ch2,o), np.kron(ch3,o)),axis = 2), dtype = original_dtype)

    def downsample_subsample(self, img, scale, cell_x = None, cell_y = None):
        if cell_x == None:
            cell_x = (scale + 1) // 2
        if cell_y == None:
            cell_y = (scale + 1) // 2
        assert(cell_x < scale)
        assert(cell_y < scale)
        return(img[cell_x::scale, cell_y::scale, :])

    def downsample_neighbor_avg(self, img, scale):
        dim_y, dim_x, _ = img.shape
        pad_x = (-dim_x) % scale
        pad_y = (-dim_y) % scale

        # cv2.copyMakeBorder(img, top, bottom, left, right, borderType = cv2.BORDER_REFLECT_101)
        padded = cv2.copyMakeBorder(img, 0, pad_x, 0, pad_y, cv2.BORDER_DEFAULT)

        # update dimensions
        dim_y = pad_y + dim_y
        dim_x = pad_x + dim_x

        # repetition count on x, y
        rep_y = int(dim_y / scale)
        rep_x = int(dim_x / scale)

        # h_ingredient
        hi1 = np.zeros((scale, scale))
        hi2 = np.zeros((scale, scale))
        hi1[0] = np.ones(scale)/scale
        hi2[:,0] = np.ones(scale)/scale

        h1 = np.kron(np.eye(rep_y), hi1)
        h2 = np.kron(np.eye(rep_x), hi2) 
        ch1 = padded[:,:,0]
        ch2 = padded[:,:,1]
        ch3 = padded[:,:,2]
        process_ch = lambda ch:np.matmul(h1,np.matmul(ch,h2))[::scale,::scale]
        avg_subsampled = np.stack((process_ch(ch1), process_ch(ch2), process_ch(ch3)), axis = 2)
        return avg_subsampled

    def downsample_gaussian(self, img, scale):
        kernel_dim = scale + (1 - scale%2)
        return(cv2.GaussianBlur(img, (kernel_dim, kernel_dim), -1)[(scale+1)//2::scale, (scale+1)//2::scale, :])

    def upscale_dnn(self, img:np.ndarray):
        return self._model.upsample(img)

    def scale_interpolation(self, img, scale, interpolation = cv2.INTER_CUBIC):
        """cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX,  """
        dsize = (img.shape[1] * scale, img.shape[0] * scale)
        return(cv2.resize(img, dsize, fx = scale, fy = scale, interpolation = interpolation))

    def upscale_nn_list(self, img_list, scale = 4) -> list:
        return [self.upscale_nearest_neighbor(img,scale) for img in img_list]

    def scale_interpolation_list(self, img_list, scale = 4, interpolation = cv2.INTER_CUBIC) -> list:
        return [self.scale_interpolation(img, scale, interpolation) for img in img_list]

    def upscale_dnn_list(self, img_list, model:str = None, scale:int = 4, verbose:bool = False):
        swap_model = not model.lower() == None and not self.get_model_n_scale() == (model, scale)
        if swap_model:
            old = self.get_model_n_scale()
            try:
                self.read_set_model(model, scale, verbose = verbose)
            except e:
                print('The model "{0}_{1}.pb" couldn`t be loaded'.format(model.upper(),scale))
                print(e)
                return img_list
        print("Upscaling {} images".format(len(img_list)))
        if verbose:
            results = []
            for img_num in range(len(img_list)):
                print("image {:3d}/{:3d} being processed       ".format(img_num+1,len(img_list)),end= "\r")
                results.append(self._model.upsample(img_list[img_num]))
            print()
        else:
            results = [self._model.upsample(img) for img in img_list]
        if swap_model  and old[0]:
            self.read_set_model(old[0],old[1])
        print("-----------------------------------------")
        return results
    
    def fig_image_grid(self, img_list, target_ratio = 16/9.0, num = None) -> matplotlib.figure.Figure:
        num_images = len(img_list)
        grid_height, grid_width = self._grid_dimensions(num_images, target_ratio)

        # create new window and display images in subplots
        if num == None:
            fig = plt.figure()
        else:
            fig = plt.figure(num = num)
            fig.suptitle(num)
        for img in range(num_images):
            plt.subplot(int(grid_height), int(grid_width), img+1)
            plt.axis("off")
            plt.margins(0)
            plt.imshow(img_list[img][:,:,::-1])
        return(fig)
          
    def fig_comp_grid(self, img_list_1, img_list_2, axis = 0, target_ratio = 16/9.0, num = None) -> matplotlib.figure.Figure:
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
        lists[small] = self.upscale_nn_list(lists[small], scale)
        comp = [np.concatenate((lists[0][i], lists[1][i]),axis = axis) for i in range(num_elem)]
        return self.fig_image_grid(comp, target_ratio, num)

    def fig_comp_dnn_orig(self, img_list, axis = 1, target_ratio = 16/9.0, model:str = None, scale:int = 4, verbose = False, num = None):
        upscaled = self.upscale_dnn_list(img_list, model, scale, verbose)
        return self.fig_comp_grid(img_list, upscaled, axis, target_ratio, num)

def load_inputs(base_path= "./", image_dir = "input_images"):
        image_extensions = [".jpg",".png"]
        image_dir = base_path + image_dir
        all_names = os.listdir(image_dir)
        image_names = [file_name for file_name in all_names if file_name[-4:] in image_extensions]
        images = [cv2.imread(image_dir +"/"+ img_name) for img_name in image_names]
        return images

if(__name__ == "__main__"):
    maximize_figs=[]
    ut = UpscaleTester()
    images = load_inputs()

    photos = images[:7]
    pixel_arts = images[7:]
    cropped = ut.crop_img_list(photos, x_start = 13/32, height = 65, y_start = 29/50, width = 110)

# Show original images
if(__name__ == "__main__"):
    #ut.fig_image_grid(cropped)
    #plt.show()
    maximize_figs.append(ut.fig_image_grid(photos, num = "Original Photos"))
    maximize_figs.append(ut.fig_image_grid(pixel_arts, num = "Pixel Arts"))

# Show crops
if(__name__ == "__main__"): # show crops
    maximize_figs.append(ut.fig_image_grid(cropped, num = "Cropped Photos"))

# Upsample dnn crop
if(__name__ == "__main__" and False): 
    print("\nUpscaling EDSR x4");upscaled_edsr = ut.upscale_dnn_list(cropped, "edsr", 4, True)
    print("\nUpscaling ESPCN x4");upscaled_espcn = ut.upscale_dnn_list(cropped, "espcn", 4, True)
    print("\nUpscaling FSRCNN x4");upscaled_fsrcnn = ut.upscale_dnn_list(cropped, "fsrcnn", 4, True)
    print("\nUpscaling FSRCNN-small x4");upscaled_fsrcnn_s = ut.upscale_dnn_list(cropped, "fsrcnn-small", 4, True)
    print("\nUpscaling LapSRN x4");upscaled_edsr = ut.upscale_dnn_list(cropped, "LapSRN", 4, True)

    maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "EDSR"))
    maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "ESPCN"))
    maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "FSRCNN"))
    maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "FSRCNN-small"))
    maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "LapSRN"))

# Upsample Pixle_Arts
if(__name__ == "__main__" and False): 
    pixel_arts_algo = "edsr"
    pixel_arts_scale = 4
    upscaled_pixel_arts1 = ut.upscale_dnn_list(pixel_arts[:7], pixel_arts_algo, pixel_arts_scale, True)
    maximize_figs.append(ut.fig_comp_grid(pixel_arts[:7], upscaled_pixel_arts1, axis = 1, num = "SR Pixel Arts1"))
    upscaled_pixel_arts2 = ut.upscale_dnn_list(pixel_arts[7:14], pixel_arts_algo, pixel_arts_scale, True)
    maximize_figs.append(ut.fig_comp_grid(pixel_arts[7:14], upscaled_pixel_arts2, axis = 1, num = "SR Pixel Arts2"))
    upscaled_pixel_arts3 = ut.upscale_dnn_list(pixel_arts[14:], pixel_arts_algo, pixel_arts_scale, True)
    maximize_figs.append(ut.fig_comp_grid(pixel_arts[14:], upscaled_pixel_arts3, axis = 1, num = "SR Pixel Arts3"))

# Downsample - subsample, area_avg, gaussian blur - downsample
if(__name__ == "__main__" and True):
    downsampled = [ut.downsample_gaussian(img, 4) for img in photos]
    maximize_figs.append(ut.fig_image_grid(downsampled, num = "Downsampled"))
# 

# Upsample the downsampled
if(__name__ == "__main__" and True):
    restored = ut.upscale_dnn_list(downsampled, "edsr", verbose = True)
    maximize_figs.append(ut.fig_comp_grid(downsampled, restored, axis = 0, num = "Restored Comparison"))
    maximize_figs.append(ut.fig_image_grid(downsampled, num = "Restored"))

# Show pyplot figures 
if(__name__ == "__main__"):
    for fig in maximize_figs:
        fig.canvas.manager.window.showMaximized()

    plt.show()


    #ut.read_set_model("ESPCN",4)
    #fig1 = ut.fig_comp_dnn_orig(cropped[:7], axis = 0, target_ratio = 8/9)
    #fig1.suptitle("ESPCN_4")

    #resized = ut.scale_interpolation_list(cropped[:7], 4, cv2.INTER_LANCZOS4)
    #fig2 = ut.fig_comp_grid(cropped[:7], resized, axis = 0, target_ratio = 16/9)
    #fig2.suptitle("INTER_LANCZOS4")
    #plt.show()

    #print("Pixel Arts")
    #target = images[7:27]
    #ut.fig_comp_dnn_orig(target, model = "EDSR", scale = 4, target_ratio = 0.5, verbose = True)
    #plt.show()
