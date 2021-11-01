# image processing
# phyton3
# Last change: 2021-08-11
# J.H

'''
Inhalt
Klassen:
Processing - 1. Used for process visualization
Diagramm   - 2. Draws diagrams
Show       - 3. Display of images and masks                        
'''

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import os

          
class Process():
    '''  1. Used for process visualization
    - Training and validation '''
    def __init__(self, training_losses,
                validation_losses,
                learning_rate,
                gaussian=True,
                sigma=2,
                figsize=(24, 18)):
        super(Process, self).__init__()                
        self.training_losses = training_losses
        self.validation_losses = validation_losses
        self.learning_rate = learning_rate
        self.gaussian = gaussian
        self.sigma = sigma
        self.figsize = figsize

    def training_validation(self, y_lim=[1,4], fontsize=20):        
        list_len = len(self.training_losses)
        x_range = list(range(1, list_len + 1))  # number of x values
        fig = plt.figure(figsize=self.figsize)
        subfigures = fig.get_axes()
        for i, subfig in enumerate(subfigures, start=1):
            subfig.spines['top'].set_visible(False)
            subfig.spines['right'].set_visible(False)
        if self.gaussian:
            training_losses_gauss = gaussian_filter(self.training_losses, sigma=self.sigma)
            validation_losses_gauss = gaussian_filter(self.validation_losses, sigma=self.sigma)
            linestyle_original = '.'
            color_original_train = 'lightcoral'
            color_original_valid = 'lightgreen'
            color_smooth_train = 'red'
            color_smooth_valid = 'green'
            alpha = 0.25
        else:
            linestyle_original = '-'
            color_original_train = 'red'
            color_original_valid = 'green'
            alpha = 1.0
        # Plot
        plt.plot(x_range, self.training_losses, linestyle_original, color=color_original_train,\
                 label='Training', alpha=alpha)
        plt.plot(x_range, self.validation_losses, linestyle_original, color=color_original_valid,\
                 label='Validation', alpha=alpha)
                
                    
        if self.gaussian:
            plt.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)   
            plt.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
        # Diagram design
        titlesize=fontsize+2
        tickssize=fontsize-2
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.ylim(y_lim)
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.title('Training & validation loss', fontsize=titlesize)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        plt.legend(loc='upper right', fontsize=tickssize)
        #return fig    

        
class Diagramm():
    ''' 2. Draws diagrams '''
    def __init__(self,figsize=(24, 18), fontsize=20, verbose=False):
        super(Diagramm, self).__init__()
        self.figsize = figsize  # Size of the representation
        self.fontsize=fontsize  # Font size
        self.verbose=verbose    # Outputs [True, False]             

    def xy(x,y, y_lim=[0,3], title='Titel', xlabel='x', ylabel='y'):
        ''' Simple x-y diagram '''
        # Diagrammgestaltung
        titlesize=self.fontsize+2
        tickssize=self.fontsize-2
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.ylim(y_lim)                     # interval
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.title(title, fontsize=titlesize)
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        plt.legend(loc='upper right', fontsize=tickssize)
        plt.plot(x, y)

        
class Show():
    ''' 
    3. Display of images and masks
    '''
    def __init__(self, experiment='', figsize=(24, 18), fontsize=20 ,verbose=False):
        super(Show, self).__init__()
        self.experiment=experiment
        self.figsize = figsize               # Größe der Darstellung
        self.fontsize=fontsize
        self.verbose=verbose                 # Testausgaben [True, False]

    def img_mask_pair(self, img, mask, img_title='Image', mask_title='Maske', instanzen=255):
        '''
        3.1 Shows a picture-mask pair
        (Objects are shown in different colors according to the instances.)
        Input parameters:
            img - Image
            mask - associated mask
            instanzen - Number of instances to be displayed
        Display output:
            img  - Image
            mask - Mask as a false color image    
        '''
        ## Own color for background -> black for background
        #https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 127)
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[:40, :] = np.array([0, 0, 0, 1])
        own_cmap = ListedColormap(newcolors)
        ###----------------------------###
        plt.figure(self.figsize) #(30,15))
        plt.subplot(221).set_title(img_title) 
        plt.imshow(img)
        plt.subplot(222).set_title(mask_title) 
        plt.imshow(mask, rasterized=False, cmap=own_cmap, vmin=0, vmax=instanzen)
        plt.colorbar()

    def img_masks_quad(self, img, masks,
                img_title='Image',
                masks_titles=('actual mask', 'predicted mask', 'post-processed mask')):
        ''' 3.2 Shows an image with three associated masks.
        Input:
            img - Image
            masks - List consisting of three masks
            img_title - Title of the image
            masks_titles - Title of the masks
        '''
        plt.figure(self.figsize)
        plt.subplot(221).set_title(img_title)
        plt.imshow(img)
        # Achtung: Drüsen der tatsächliche Maske sind indiziert !!
        plt.subplot(222).set_title(masks_titles[0]) 
        plt.imshow(masks[0])
        # Klassifizierung (0- Hintergrund, 1- Drüsen)
        plt.subplot(223).set_title(masks_titles[1])  
        plt.imshow(masks[1])
        plt.subplot(224).set_title([2])  
        plt.imshow(masks[2])

    def list_set(self, idx_list, listset, titles, path=''):
        ''' STANDARD VERSION
        3.3 Shows an image with multiple masks and multiple samples.
        Input:
            idx_list - List of indices of the samples to be displayed.
            listset  - List of complete pictures and masks
            titels   - Title of the individual list elements
                       (Number must match number in list set.)
            path - Path for saving the images mask set.
                   (If no path is specified, there is no saving.)
        Display:
            Image set consisting of images and masks                    
        '''
        rows=len(listset)
        cols=len(idx_list)
        plt.figure(figsize=self.figsize)
        for r in range(0, rows):
            for c in range (0, cols):
                idx=idx_list[c]
                plt.subplot(rows,cols,1+r*cols + c).set_title(f'{titles[r]} {idx}',fontsize=self.fontsize)
                plt.imshow(listset[r][idx])
                plt.axis('off')
        plt.suptitle(f'Image-Masken-Set\n{os.path.basename(path)}\n{self.experiment}')
        if path != '':
            plt.savefig(path, bbox_inches="tight")
        plt.show()
    
    def histogramm(self, img, save_path='temp.png', abszisse='Id', ordinate='Häufigkeit H', relativ=True):
        '''
        3.4 Frequency distribution of the instances / intensities of a mask or an image.
        img - Image oder mask
        save_path - Path for saving the histogram
        abszisse - Name of the abscissa
        ordinate - Name of the ordinate
        relativ - relative or absolute scaling of the frequency
        '''
        max_id=img.max() # Maximum value in the mask
        plt.title("Histogram\n", fontsize=16)
        plt.xlabel(abszisse, fontsize=14)
        plt.ylabel(ordinate, fontsize=14) #density=True
        n,bins,patches=plt.hist(img.ravel(), bins=int(max_id+2), range=(0, max_id+2),\
                 histtype='bar', align='left', color='gray', density=relativ, ec='k', cumulative=False )
        plt.savefig(save_path, bbox_inches="tight")                 
        #plt.show()                 
        return n,bins,patches        


   

  





            

