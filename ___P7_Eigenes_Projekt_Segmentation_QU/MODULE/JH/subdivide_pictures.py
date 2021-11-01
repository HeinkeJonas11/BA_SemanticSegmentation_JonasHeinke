# phyton3
# Last change: 2021-06-27
# Autor: J. Heinke

''' Content:
1. SubdividePictures   - Divides large images into several small images and saves them in
                        separate directories "./subimg" and "./submask"
'''
import numpy as np
from skimage import io #imread, imsave, imshow
import os
import matplotlib.pyplot as plt
from PIL import Image

class SubdividePictures(object):
    ''' 
    1. Divides large images into several small images and saves them in
                        separate directories "./subimg" and "./submask"
    '''
    def __init__(self, image_pathnames, mask_pathnames, path_sub_images, path_sub_masks,
                out_shape=(512,512), instanzen=2, verbose=True):
        super(SubdividePictures, self).__init__()
        # Source paths for input images and masks (targets)
        self.image_pathnames = image_pathnames
        self.mask_pathnames = mask_pathnames    
        # Size of the original images
        self.out_shape=out_shape
        # Size of an original image (h, w) -> is updated
        self.in_shape=(4480, 7392)
        # Output paths for subimages (directory name also specifies the resolution of the output image masks).
        self.path_sub_images= str(path_sub_images) + str(out_shape)
        self.path_sub_masks = str(path_sub_masks)  + str(out_shape)
        # Create target paths or target directories if they do not exist.
        try:
            os.mkdir(self.path_sub_images)
            os.mkdir(self.path_sub_masks)
        except:
            print('Sub-directories exist')
        # Objects are shown in different colors according to the instances.          
        self.instanzen=instanzen-1 # -1 -> Background is not counted.
        # Output control for testing
        self.verbose=verbose


    def allpairs(self):
        ''' 1.1 For all original images and masks '''
        for idx in range(len(self.image_pathnames)):
            self.onepair(idx)
            print(idx, end=' ') # go

    def onepair(self,idx=0):
        ''' 1.2 Generates several sub-image-sub-mask pairs from an image-mask pair.
        Input parameters:
            idx - Index of directory lists for images and masks.
       Save as a file::
            sub_img  - List of subpictures  -> '/subimg/ *.png
            sub_mask - List of submasks -> '/submask/ *.png
            Note: The file names are made up of the name of the source file and
            the coordinates of the upper left corner of the sub-image
            in relation to the input picture.
        '''
        # https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread
        image=io.imread(self.image_pathnames[idx])
        mask=io.imread(self.mask_pathnames[idx])
        # Update original image size, different input image sizes are possible
        self.in_shape=mask.shape #mask_shape
        # Filters the names of the input image and the associated mask from the path names.
        image_filename=os.path.splitext(os.path.basename(self.image_pathnames[idx]))[0]
        mask_filename =os.path.splitext(os.path.basename(self.mask_pathnames[idx] ))[0]
        if self.verbose:
            # Test-Images
            img1=image[0:self.out_shape[0], 0:self.out_shape[1], :]
            mask1=mask[0:self.out_shape[0], 0:self.out_shape[1]]
            print('-- subimg ---')
            print(self.path_sub_images)
            print('-- submask ---')
            print(self.path_sub_masks)
            self.pairshow(image, mask)
            print('----------------------------')
        points=self.img_boarders()
        for p in points:
            x1=p[0]
            y1=p[1]
            x2=p[0]+ self.out_shape[0]
            y2=p[1]+ self.out_shape[1]  
            sub_img= image[x1:x2, y1:y2, :]
            sub_mask=mask[x1:x2, y1:y2]
            # Create path with filename of a sub-image-mask-pair
            sum_image_filepath=str(self.path_sub_images) + '/'+ image_filename + '_x' +str(x1)+ 'y'+str(y1)+'.png'
            sum_mask_filepath=str(self.path_sub_masks) + '/' + mask_filename + '_x'+str(x1)+ 'y'+str(y1)+'.png'
            # Save sub-image-masks
            io.imsave(sum_image_filepath, sub_img)
            io.imsave(sum_mask_filepath, sub_mask) #imsave() got multiple values for argument 'arr'
            # Control issue
            if self.verbose:
                self.pairshow(sub_img, sub_mask)

    def pairshow(self, img, mask):
        ''' HELP METHOD
        1.3 Shows a picture-mask pair
        Input parameters:
            img -  Image
            mask - Associated mask
        Display output:
            img  - Image
            mask - Mask as a false color image
        '''
        # own color for background -> black
        # https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 127)
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[:40, :] = np.array([0, 0, 0, 1])
        own_cmap = ListedColormap(newcolors)
        plt.figure(figsize=(30,15))
        plt.subplot(221) 
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(mask, rasterized=False, cmap=own_cmap, vmin=0, vmax=self.instanzen)
        plt.colorbar()


    def img_boarders(self):
        '''
        1.4 Calculates the points of the upper left corner of the sub-picture masks of an input image.
        Input parameters:
           Class / instance variables
        Return:
            points - List with the points of the sub-picture masks.          
        '''
        # Number vertically and horizontally
        count  = (self.in_shape[0] // self.out_shape[0] + 1,
                self.in_shape[1] // self.out_shape[1] + 1)
        # Total overlap because the pitch is uneven              
        overlap= (self.out_shape[0]*count[0] - self.in_shape[0],
                self.out_shape[1]*count[1] - self.in_shape[1])
        # Overlap for two adjacent images
        offset= (overlap[0] // (count[0]-1),
                overlap[1] // (count[1]-1))
        # Points list
        points=[]
        # Calculation of the corner points 
        # - last column and last row are calculated differently.
        for i in range(0, count[0]-1):
            for j in range(0, count[1]-1):
                point=(i * (self.out_shape[0]- offset[0]),
                       j * (self.out_shape[1]- offset[1]))
                points.append(point)
        # right margin images
        for i in range( 0,  count[0]-1):
            point= (i * (self.out_shape[0]- offset[0]),
                    self.in_shape[1] - self.out_shape[1])
            points.append(point)    
        # lower edge images
        for j in range( 0, count[1]-1):
            point=(self.in_shape[0] - self.out_shape[0],
                   j * (self.out_shape[1]- offset[1]))
            points.append(point)
        # Right picture, below
        point=(self.in_shape[0] - self.out_shape[0],
               self.in_shape[1] - self.out_shape[1])
        points.append(point)
        if self.verbose:
            # Control issue
            print (count)
            print (overlap)
            print(offset)
            print(points)
        return points            

    


   
        
	


