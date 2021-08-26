print("make sure to install dependencies\npip install numpy opencv matplotlib")
print("> Importing openCV                 ",end = "\r")
import matplotlib.pyplot as plt
print("> Importing numpy                  ",end = "\r")
import numpy as np
import numpy.linalg as LA
print("> Importing openCV                 ",end = "\r")
import cv2
import os
print("                                   ")
import matplotlib
from math import log10, sqrt

from skimage.metrics import structural_similarity as ssim
from skimage import color, data, restoration

from enum import Enum

class UpscaleTester:
    """
    base_dir: Directory where the code is being executed
    output_dir: an output directory inside the base directory
    _model: cv2.dnn_superres model. the current model set for the class instance

    only uses CPU
    """
    def __init__(self, base_dir = "./"):
        """
        initialize with user input base_dir
        rename the downloaded models in the /model folder so this code can recognize them
        """
        self.base_dir = base_dir
        self.output_dir = base_dir + "output_images"
        if not os.path.isdir(self.output_dir+"/Cropped"):
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            os.mkdir(self.output_dir+"/Cropped")
        self._model = cv2.dnn_superres.DnnSuperResImpl_create()
        self._rename_models_files()
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
   
    def _rename_models_files(self):
        if not os.path.exists(self.base_dir+"/models"):
            os.mkdir(self.base_dir+"/models")
        for file_name in os.listdir(self.base_dir+"/models"):
            if file_name[-3:].lower() == ".pb":
                os.rename(self.base_dir+"/models/"+file_name, self.base_dir+"/models/"+file_name[:-6].upper()+file_name[-6:])
        
    def _grid_dimensions(self, num_elem:int, target_ratio:float = (16/9.0)) -> (int, int):
        """
        num_elem: How many images do you want to display in subplot grid?
        target_ratio: What is your target aspect ration?
        """
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
        """
        Take in a list of images and return list of images 
        that starts from the designated part inside the image,
        and end at certain point or after a certain pixels
        """
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
        """
        model: String of the name of the dnn model the user want to use for upscaling
        scale: integer number for the model mostly x2, x3, x4 is available and LapSRN uses x8
        
        The corresponding model files must have been downloaded into 
        the /models directory and renamed by initializing this class
        """
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
        """
        Returns information of current model and scale
        """
        model_name = self._model.getAlgorithm().lower()
        scale_value = self._model.getScale()
        return(model_name, scale_value)

    def downsample_subsample(self, img, scale, cell_x = None, cell_y = None):
        """
        Downsample an image by picking out exact pixels in uniform intervals
        """
        if cell_x == None:
            cell_x = (scale + 1) // 2
        if cell_y == None:
            cell_y = (scale + 1) // 2
        assert(cell_x < scale)
        assert(cell_y < scale)
        return(img[cell_x::scale, cell_y::scale, :])

    def downsample_neighbor_avg(self, img, scale):
        """
        Divide the original image into grids, each cell contating scale x scale number of pixels,
        average each cells and use them to make x_scale downsampled image
        Implemented with kronecker multiplication and matrix multiplication
        """
        dim_x, dim_y, _ = img.shape
        pad_x = (-dim_x) % scale
        pad_y = (-dim_y) % scale

        # cv2.copyMakeBorder(img, top, bottom, left, right, borderType = cv2.BORDER_REFLECT_101)
        padded = cv2.copyMakeBorder(img, pad_x//2, (pad_x+1)//2, pad_y//2, (pad_y+1)//2, cv2.BORDER_REFLECT)

        # update dimensions
        dim_x = pad_x + dim_x
        dim_y = pad_y + dim_y

        # repetition count on x, y
        rep_x = int(dim_x / scale)
        rep_y = int(dim_y / scale)

        # h_ingredient
        hi1 = np.zeros((scale, scale))
        hi2 = np.zeros((scale, scale))
        hi1[0] = np.ones(scale)/scale
        hi2[:,0] = np.ones(scale)/scale

        h1 = np.kron(np.eye(rep_x), hi1)
        h2 = np.kron(np.eye(rep_y), hi2) 
        ch1 = padded[:,:,0]
        ch2 = padded[:,:,1]
        ch3 = padded[:,:,2]
        process_ch = lambda ch:np.matmul(h1,np.matmul(ch,h2))[::scale,::scale]
        avg_subsampled = np.stack((process_ch(ch1), process_ch(ch2), process_ch(ch3)), axis = 2).astype(int)
        return avg_subsampled

    def downsample_gaussian(self, img, scale):
        """
        subsample after putting gaussian filter on the image
        """
        kernel_dim = scale + (1 - scale%2)
        return(cv2.GaussianBlur(img, (kernel_dim, kernel_dim), -1)[(scale)//2::scale, (scale)//2::scale, :])

    def upscale_dnn(self, img):
        """
        Use self.modle to upscale the image
        """
        return self._model.upsample(img)

    def scale_interpolation(self, img, scale, interpolation = cv2.INTER_CUBIC):
        """cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4, cv2.INTER_LINEAR_EXACT, cv2.INTER_NEAREST_EXACT, cv2.INTER_MAX,  """
        dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        return(cv2.resize(img, dsize, fx = scale, fy = scale, interpolation = interpolation))

    def scale_interpolation_list(self, img_list, scale = 4, interpolation = cv2.INTER_CUBIC) -> list:
        """
        self.scale_interpolation() for every images in the list, returns a list of the results
        """
        return [self.scale_interpolation(img, scale, interpolation) for img in img_list]

    def upscale_dnn_list(self, img_list, model:str = None, scale:int = 4, verbose:bool = False):
        """
        Upscale all the images in the list using dnn.
        model can be specified or not. 
            If not specified, it will use self.model 
            If specified, it will be used temporarily and swapped back to the original model the class was set with
        """
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
        """
        Display images in grid format, best arrangement to fit the target aspect ratio
        """
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
        stacks two images in each img_list. If the dimensions doesn't match, it is upscaled with INTER_NEAREST to match.

        The two lists' elements should match each other, and the scale difference should be consistent.
        axis = 0 : vertical concatenation
        axis = 1 : horizontal concatenation
        """
        
        num_elem = len(img_list_1)
        if not len(img_list_2) == num_elem:
            raise Exception("Number of images doesn't match for the two lists\nimg_list_1: {0}  img_list_2: {1}".format(num_elem, len(img_list_2)))
        lists = [img_list_1.copy(), img_list_2.copy()]
        dims = [ [img.shape[1::-1] for img in lists[0]], [img.shape[1::-1] for img in lists[1]] ]
        
        for i in range(num_elem):
            bigger = 0 if dims[0][i][0]>dims[1][i][0] else 1
            smaller = 1 - bigger
            lists[smaller][i] = cv2.resize(lists[smaller][i], dims[bigger][i], interpolation = cv2.INTER_NEAREST)
        comp = [np.concatenate((lists[0][i], lists[1][i]), axis = axis) for i in range(num_elem)]
        return self.fig_image_grid(comp, target_ratio, num)

    def fig_comp_dnn_orig(self, img_list, axis = 1, target_ratio = 16/9.0, model:str = None, scale:int = 4, verbose = False, num = None):
        """
        compares dnn upscaled image with the original image
        """
        upscaled = self.upscale_dnn_list(img_list, model, scale, verbose)
        return self.fig_comp_grid(img_list, upscaled, axis, target_ratio, num)

    def show_full_figures(self, img_list):
        for imgs in img_list:
            plt.figure()
            plt.imshow(imgs[:,:,::-1])

    def calculate_mse(self, img1, img2, crop_or_resize = False, best = True):
        """
        take to images, that might have slightly disagreeing dimensions and calculate mse
        """
        imgs = [img1, img2]
        dims = (img1.shape[:2], img2.shape[:2])
        bigger = 0 if dims[0][0]>dims[1][0] else 1
        if(abs(dims[0][0]/dims[1][0]-1) > 0.05 or abs(dims[0][1]/dims[1][1]-1) > 0.05):
            print("The two images are too different in size:\n{0},{1}".format(dims[0],dims[1]))
        smaller = 1 - bigger

        small = imgs[smaller] 
        if best:
            big = imgs[bigger][:dims[smaller][0],:dims[smaller][1],:]
            assert(big.shape == small.shape)
            mse_1 = np.mean((small-big)**2)

            big = cv2.resize(imgs[bigger],dims[smaller][1::-1])
            assert(big.shape == small.shape)
            mse_2 = np.mean((small-big)**2)
            mse_ = mse_1 if mse_1 < mse_2 else mse_2
        else:
            if crop_or_resize:
                big = imgs[bigger][:dims[smaller][0],:dims[smaller][1],:]
            else:
                big = cv2.resize(imgs[bigger],dims[smaller][1::-1], interpolation = cv2.INTER_NEAREST)
            assert(big.shape == small.shape)
            mse_ = np.mean((big-small)**2)
        return(mse_)

    def calculate_mse_list(self, img_list_1, img_list_2, crop_or_resize = False, best = True):
        mse_list = []
        num_elems = len(img_list_1)
        assert(num_elems==len(img_list_2))
        for ii in range(num_elems):
            mse_list.append(self.calculate_mse(img_list_1[ii], img_list_2[ii], crop_or_resize, best))
        return(mse_list)

    def calculate_PSNR(self, img1, img2, max_pixel = 255):
        return 20 * log10(max_pixel / sqrt(self.calculate_mse(img1,img2)))

    def calculate_PSNR_list(self, img_list_1, img_list_2):
        psnr_list = []
        num_elems = len(img_list_1)
        for ii in range(num_elems):
            psnr_list.append(self.calculate_PSNR(img_list_1[ii], img_list_2[ii]))
        return(psnr_list)

    def calculate_ssim(self,img1, img2):
        img1, img2 = self._trim_identical_images(img1, img2)
        return ssim(img1, img2, multicahnnel = True)

    def calculate_ssim_list(self, img_list_1, img_list_2):
        num_elems = len(img_list_1)
        return [ssim(img_list_1[ii], img_list_2[ii], multichannel = True) for ii in range(num_elems)]

    def wiener_deconv(self, img):
        ch1 = img[:, :, 0]
        ch2 = img[:, :, 1]
        ch3 = img[:, :, 2]
        channels = [restoration.unsupervised_wiener(mono) for mono in (ch1, ch2, ch3)]
        image_ = np.stack(channels, axis = 2)
        plt.imshow(image_)
        plt.show()
        return image_

    def lucy_deconv(self, img):
        pass

    def save_images(self, img_list, dir_name, prefix = None):
        if prefix == None:
            prefix = dir_name
        for img_num in range(len(img_list)):
            cv2.imwrite(ut.output_dir+"/"+dir_name+"/"+prefix+"_{:03d}.jpg".format(img_num+1), img_list[img_num])

class ds_opt(Enum):
    AREA = 1
    CUBIC = 2
    LINEAR = 3
    NEAREST = 4
    GAUSSIAN = 5
    GRID_AVG = 6

def load_inputs(base_path= "./", image_dir = "input_images"):
        image_extensions = [".jpg",".png"]
        image_dir = base_path + image_dir
        all_names = os.listdir(image_dir)
        image_names = [file_name for file_name in all_names if file_name[-4:] in image_extensions]
        images = [match_dim24(cv2.imread(image_dir +"/"+ img_name)) for img_name in image_names]
        return images

def match_dim24(img):
    dim = img.shape[1::-1]

    pad_horizontal = (dim[0]%24)/dim[0] > 0.08
    pad_vertical = (dim[1]%24)/dim[1] > 0.08

    horizontal_padding = (-dim[0])%24 if pad_horizontal else 0
    vertical_padding = (-dim[1])%24 if pad_vertical else 0

    img = cv2.copyMakeBorder(img, vertical_padding//2, (vertical_padding+1)//2, horizontal_padding//2, (horizontal_padding+1)//2, borderType = cv2.BORDER_CONSTANT, value = (255, 255, 255))

    padded_dim24 = [dim[0] + horizontal_padding, dim[1] + vertical_padding]

    return img[:padded_dim24[1] - padded_dim24[1]%24, :padded_dim24[0] - padded_dim24[0]%24,::]

grid_original_images = False
grid_crop_images = True
photos_all_dnns = False
pixel_arts_dnn = True
downsample_photos = True
downsampling_method = ds_opt.AREA

full_dnn = False
full_dnn_method = "fsrcnn"
grid_full_dnn_cropped = False
comp_full_dnn_cropped = False
full_dnn_extra = False

dnn_cropped = False
dnn_cropped_method = "espcn" 
grid_cropped_dnn = True
comp_cropped_dnn = True

resize_upsample = False
grid_resize_crop = False
comp_resize_crops= False
grid_resize = False
singles_cubic_sr = False
singles_lanczos_sr = False

x_start = 13/32; height = 68; y_start = 29/50; width = 112

photos_index_end = 114

ds_scale = 4
pixel_arts_algo = "edsr"
pixel_arts_scale = 4

output_scale_string = "x{}".format(ds_scale)

if(__name__ == "__main__"):
    maximize_figs=[]
    ut = UpscaleTester()
    if not os.path.isdir(ut.output_dir+"/"+output_scale_string):
        os.mkdir(ut.output_dir+"/"+output_scale_string)
    images = load_inputs()
    ut.save_images(images, output_scale_string+"/Original", "Original")

    photos = images[:photos_index_end]
    pixel_arts = images[photos_index_end:]
    cropped = ut.crop_img_list(photos, x_start = x_start, height = height, y_start = y_start, width = width)
    ut.save_images(cropped, "Cropped/Original", "Original_Crop")

    # Show original images
    if(grid_original_images):
        maximize_figs.append(ut.fig_image_grid(photos, num = "Original Photos"))
        maximize_figs.append(ut.fig_image_grid(pixel_arts, num = "Pixel Arts"))

    # Show crops
    if(grid_crop_images): # show crops
        maximize_figs.append(ut.fig_image_grid(cropped, num = "Cropped Photos"))

    # Upsample dnn crop
    if(photos_all_dnns): 
        print("\nUpscaling EDSR x4");upscaled_edsr = ut.upscale_dnn_list(cropped, "edsr", ds_scale, True)
        print("\nUpscaling ESPCN x4");upscaled_espcn = ut.upscale_dnn_list(cropped, "espcn", ds_scale, True)
        print("\nUpscaling FSRCNN x4");upscaled_fsrcnn = ut.upscale_dnn_list(cropped, "fsrcnn", ds_scale, True)
        print("\nUpscaling FSRCNN-small x4");upscaled_fsrcnn_s = ut.upscale_dnn_list(cropped, "fsrcnn-small", ds_scale, True)
        print("\nUpscaling LapSRN x4");upscaled_edsr = ut.upscale_dnn_list(cropped, "LapSRN", ds_scale, True)

        maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "EDSR"))
        maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "ESPCN"))
        maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "FSRCNN"))
        maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "FSRCNN-small"))
        maximize_figs.append(ut.fig_comp_grid(cropped, upscaled_edsr, num = "LapSRN"))

    # Upsample Pixle_Arts
    if(pixel_arts_dnn): 
        upscaled_pixel_arts1 = ut.upscale_dnn_list(pixel_arts[:7], pixel_arts_algo, pixel_arts_scale, True)
        maximize_figs.append(ut.fig_comp_grid(pixel_arts[:7], upscaled_pixel_arts1, axis = 1, num = "SR Pixel Arts1"))
        upscaled_pixel_arts2 = ut.upscale_dnn_list(pixel_arts[7:14], pixel_arts_algo, pixel_arts_scale, True)
        maximize_figs.append(ut.fig_comp_grid(pixel_arts[7:14], upscaled_pixel_arts2, axis = 1, num = "SR Pixel Arts2"))
        upscaled_pixel_arts3 = ut.upscale_dnn_list(pixel_arts[14:20], pixel_arts_algo, pixel_arts_scale, True)
        maximize_figs.append(ut.fig_comp_grid(pixel_arts[14:20], upscaled_pixel_arts3, axis = 1, num = "SR Pixel Arts3"))
        if not os.path.isdir(ut.output_dir+"/Pixel Arts"):
            os.mkdir(ut.output_dir+"/Pixel Arts")
        ut.save_images(upscaled_pixel_arts1 + upscaled_pixel_arts2 + upscaled_pixel_arts3, "Pixel Arts/x{}".format(pixel_arts_scale), "Pixel Arts HR({} x{})".format(pixel_arts_algo, pixel_arts_scale))

    # Downsample - Area, Cubic, Linear, Nearest, Gaussian, Grid_Avg
    if(downsample_photos):
        downsampled = ut.scale_interpolation_list(photos, scale = 1/ds_scale, interpolation = cv2.INTER_AREA)
        ut.save_images(downsampled, output_scale_string+"/LR", "LR(x{:d})".format(ds_scale))
    
        if downsampling_method == ds_opt.AREA:
            downsampled_crop = ut.scale_interpolation_list(cropped, scale = 1/ds_scale, interpolation = cv2.INTER_AREA)
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "Cropped Downsampled AREA"))
        if downsampling_method == ds_opt.CUBIC:
            downsampled_crop = ut.scale_interpolation_list(cropped, scale = 1/ds_scale, interpolation = cv2.INTER_CUBIC)
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "Cropped Downsampled CUBIC"))
        if downsampling_method == ds_opt.LINEAR:
            downsampled_crop = ut.scale_interpolation_list(cropped, scale = 1/ds_scale, interpolation = cv2.INTER_LINEAR)
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "Cropped Downsampled LINEAR"))
        if downsampling_method == ds_opt.NEAREST:
            downsampled_crop = ut.scale_interpolation_list(cropped, scale = 1/ds_scale, interpolation = cv2.INTER_NEAREST)
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "Cropped Downsampled NEAREST"))
        if downsampling_method == ds_opt.GAUSSIAN:
            downsampled_crop = [ut.downsample_gaussian(img, ds_scale) for img in cropped]
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "Cropped Downsampled gaussian"))
        if downsampling_method == ds_opt.GRID_AVG:
            downsampled_crop = [ut.downsample_neighbor_avg(img,ds_scale) for img in cropped]
            maximize_figs.append(ut.fig_image_grid(downsampled_crop, num = "neighbor average"))
        ut.save_images(downsampled_crop, output_scale_string+"/Cropped/"+output_scale_string+"/R {}".format(downsampling_method), "LR_Crop({0} {1})".format(downsampling_method, ds_scale))

    # Upsample the full downsampled
    if(full_dnn):
        restored = ut.upscale_dnn_list(downsampled, full_dnn_method, scale = ds_scale ,verbose = True)
        cropped_restored = ut.crop_img_list(restored, x_start = x_start, height = height, y_start = y_start, width = width)
        
        ut.save_images(restored, output_scale_string+"/R {}".format(full_dnn_method.upper()), "Restored({} x{})".format(full_dnn_method, ds_scale))
        ut.save_images(cropped_restored, "/Cropped/"+output_scale_string+"/R {}".format(full_dnn_method.upper()), "Restored_Crop({} x{})".format(full_dnn_method, ds_scale))
        print("Restored Full")
        #print("MSE", ut.calculate_mse_list(restored, photos))
        print(np.mean(ut.calculate_mse_list(restored, photos)))
        #print("PSNR", ut.calculate_PSNR_list(restored, photos))
        print(np.mean(ut.calculate_PSNR_list(restored, photos)))
        #print("SSIM", ut.calculate_ssim_list(restored, photos))
        print(np.mean(ut.calculate_ssim_list(restored, photos)))

        #print("Cropped Restored")
        #print("MSE", ut.calculate_mse_list(cropped_restored, cropped))
        #print("PSNR", ut.calculate_PSNR_list(cropped_restored, cropped))
        if(grid_full_dnn_cropped):
            maximize_figs.append(ut.fig_image_grid(cropped_restored, num = "Cropped Restored"))
        if(comp_full_dnn_cropped):
            maximize_figs.append(ut.fig_comp_grid(cropped, cropped_restored, axis = 0, num = "Cropped Restored Comparison"))
        if(full_dnn_extra):
            maximize_figs.append(ut.fig_image_grid(restored, num = "Restored"))
            ut.show_full_figures(restored)

    # Upsample the cropped downsampled
    if(dnn_cropped):
        restored_crop = ut.upscale_dnn_list(downsampled_crop, dnn_cropped_method, scale = ds_scale, verbose = True)
    
        print("Restored Crop")
        print("MSE", ut.calculate_mse_list(restored_crop, cropped))
        print("PSNR", ut.calculate_PSNR_list(restored_crop, cropped))
        if(grid_cropped_dnn):
            maximize_figs.append(ut.fig_image_grid(restored_crop, num = "Restored crops"))
        if(comp_cropped_dnn):
            maximize_figs.append(ut.fig_comp_grid(cropped, restored_crop, axis = 0, num = "Restored crops Comparison"))

    # Upsample with resizing
    if(resize_upsample):
        resized_cubic = ut.scale_interpolation_list(downsampled, scale = ds_scale, interpolation = cv2.INTER_CUBIC)
        resized_lanczos = ut.scale_interpolation_list(downsampled, scale = ds_scale, interpolation = cv2.INTER_LANCZOS4)
        resized_nearest = ut.scale_interpolation_list(downsampled, scale = ds_scale, interpolation = cv2.INTER_NEAREST)

        ut.save_images(resized_cubic, output_scale_string + "/R BICUBIC", "Bicubic(x{})".format(ds_scale))
        ut.save_images(resized_lanczos, output_scale_string + "/R LANCZOS", "Lanczos(x{})".format(ds_scale))
        ut.save_images(resized_nearest, output_scale_string + "/LR NEAREST", "LR - Nearest(x{})".format(ds_scale))

        cropped_cubic = ut.crop_img_list(resized_cubic, x_start = x_start, height = height, y_start = y_start, width = width)
        cropped_lanczos = ut.crop_img_list(resized_lanczos, x_start = x_start, height = height, y_start = y_start, width = width)
        cropped_nearest = ut.crop_img_list(resized_nearest, x_start = x_start, height = height, y_start = y_start, width = width)

        ut.save_images(cropped_cubic, "/Cropped/"+output_scale_string+"/R BICUBIC", "Bicubic(x{})".format(ds_scale))
        ut.save_images(cropped_lanczos, "/Cropped/"+output_scale_string+"/R LANCZOS", "Lanczos(x{})".format(ds_scale))
        ut.save_images(cropped_nearest, "/Cropped/"+output_scale_string+"/LR NEAREST", "LR - Nearest(x{})".format(ds_scale))

        print("Cubic resize Full")
        #print("MSE", ut.calculate_mse_list(resized_cubic, photos))
        print(np.mean(ut.calculate_mse_list(resized_cubic, photos)))
        #print("PSNR", ut.calculate_PSNR_list(resized_cubic, photos))
        print(np.mean(ut.calculate_PSNR_list(resized_cubic, photos)))
        #print("SSIM", ut.calculate_ssim_list(resized_cubic, photos))
        print(np.mean(ut.calculate_ssim_list(resized_cubic, photos)))
        print("Lanczos resize Full")
        #print("MSE", ut.calculate_mse_list(resized_lanczos, photos))
        print(np.mean(ut.calculate_mse_list(resized_lanczos, photos)))
        #print("PSNR", ut.calculate_PSNR_list(resized_lanczos, photos))
        print(np.mean(ut.calculate_PSNR_list(resized_lanczos, photos)))
        #print("SSIM", ut.calculate_ssim_list(resized_cubic, photos))
        print(np.mean(ut.calculate_ssim_list(resized_cubic, photos)))

        if(grid_resize_crop):
            maximize_figs.append(ut.fig_image_grid(cropped_cubic, num = "Cropped - resized cubic"))
            maximize_figs.append(ut.fig_image_grid(cropped_lanczos, num = "Cropped - resized lanczos"))
        if(comp_resize_crops):
            maximize_figs.append(ut.fig_comp_grid(cropped, cropped_cubic, axis = 0, num = "Cropped cubic Comparison"))
            maximize_figs.append(ut.fig_comp_grid(cropped, cropped_lanczos, axis = 0, num = "Cropped lanczos Comparison"))
        if(grid_resize):
            maximize_figs.append(ut.fig_image_grid(resized_cubic, num = "Resized cubic"))
            maximize_figs.append(ut.fig_image_grid(resized_lanczos, num = "Resized lanczos"))
        if(singles_cubic_sr):
            ut.show_full_figures(resized_cubic)
        if(singles_lanczos_sr):
            ut.show_full_figures(resized_lanczos)

    # Show pyplot figures 
    for fig in maximize_figs:
        fig.canvas.manager.window.showMaximized()

    print("esrgans")
    esrgans=load_inputs(r"D:\Dropbox\others\Real-ESRGAN\results\x2",".")
    plt.figure()
    plt.imshow(esrgans[-1])
    #maximize_figs.append(ut.fig_comp_grid(esrgans, photos, axis = 0, num = "esrgans"))
    #print(np.mean(ut.calculate_mse_list(esrgans, photos)))
    #print(np.mean(ut.calculate_PSNR_list(esrgans, photos)))
    #print(np.mean(ut.calculate_ssim_list(esrgans, photos)))

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
