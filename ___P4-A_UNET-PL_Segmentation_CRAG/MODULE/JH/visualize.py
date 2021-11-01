# image processing
# phyton3
# Letzte Änderung: 2021-04-
# J.H
# Unter Verwendung der Volrage von JS.

'''
Inhalt
Klassen:
Processing - 1. Dient zur Prozessvisualisierung
Diagramm   - 2. Erstellt x-y-Diagramm
Show       - 2. Dient zur Darstellung von Bildern und Masken

                          
'''

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import os

          
class Process():
    '''  1. Dient zur Prozessvisualisierung
    - Training, Validierung '''
    
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
        plt.plot(x_range, self.training_losses, linestyle_original, color=color_original_train, label='Training',
                    alpha=alpha)
        plt.plot(x_range, self.validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                    alpha=alpha)
        if self.gaussian:
            plt.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
            plt.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)

        # Diagrammgestaltung
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
    ''' Zeichnet Diagramme '''
    def __init__(self,figsize=(24, 18), fontsize=20, verbose=False):
        super(Diagramm, self).__init__()
        self.figsize = figsize               # Größe der Darstellung
        self.fontsize=fontsize
        self.verbose=verbose                 # Testausgaben [True, False]

    def xy(x,y, y_lim=[0,3], title='Titel', xlabel='x', ylabel='y'):
        ''' Einfaches x-y-Diagramm'''
        # Diagrammgestaltung
        titlesize=self.fontsize+2
        tickssize=self.fontsize-2
        plt.xticks(fontsize=tickssize)
        plt.yticks(fontsize=tickssize)
        plt.ylim(y_lim)
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.title(title, fontsize=titlesize)
        plt.xlabel(xlabel, fontsize=self.fontsize)
        plt.ylabel(ylabel, fontsize=self.fontsize)
        plt.legend(loc='upper right', fontsize=tickssize)
        plt.plot(x, y)


# ---------------------------------------------------------------------

        
class Show():
    ''' 
    2. Dient zur Darstellung von Bildern und Masken
    '''
    def __init__(self, experiment='', figsize=(24, 18), fontsize=20 ,verbose=False):
        super(Show, self).__init__()
        self.experiment=experiment
        self.figsize = figsize               # Größe der Darstellung
        self.fontsize=fontsize
        self.verbose=verbose                 # Testausgaben [True, False]

    def img_mask_pair(self, img, mask, img_title='Image', mask_title='Maske', instanzen=255):
        '''
        Zeigt ein Bild-Masken-Paar.
        (Objekte werden entsprechend der Instanzen unterschiedlich farblich dargestellt.)
        Eingangsparameter:
            img - Bild bzw. Image
            mask - zugehörige Maske
            instanzen - Anzahl der darzustellenden Instanzen
        Displayausgabe:
            img  - Bild
            mask - Maske als Falschfarbenbild    
        '''
        ## eigene Farbe für Hintergrund -> Schwarz für Hintergrund
        #https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 127)
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[:40, :] = np.array([0, 0, 0, 1])
        own_cmap = ListedColormap(newcolors)
        ###----------------------------###
        #print(img.shape, mask.shape)
        plt.figure(self.figsize) #(30,15))
        plt.subplot(221).set_title(img_title) 
        plt.imshow(img)
        #plt.colorbar()
        plt.subplot(222).set_title(mask_title) 
        plt.imshow(mask, rasterized=False, cmap=own_cmap, vmin=0, vmax=instanzen)
        plt.colorbar()

    def img_masks_quad(self, img, masks,
        img_title='Image',
        masks_titles=('tatsächliche Maske', 'prognostizierte Maske', 'nachbearbeitete Maske')):
        ''' Zeigt ein Bilder mit drei zugehörigen Masken
        img - Image, Bild
        masks - Liste, bestehend aus drei Masken
        img_title - Titel des Bildes
        masks_titles - Titel der masken
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


    def list_set_col(self, samples, fnameimage, fnamemask):
        # NICHT GETESTET
        '''
            Zeigt zeilennweise ein Image mit den Masken für drei Samples,
            zusätzlich mit Binaer-Intensitaet-Morphology (opening),
        '''
        for col in range(3):
            for row in range(3):
                #index = val_set_idx[samples[row]]
                ax[row, 0].set_title(f'Image {samples[row]}')
                ax[row, 1].set_title(f'tatsächliche Maske {samples[row]}')
                ax[row, 2].set_title(f'prognostizierte Maske {samples[row]}')
                
                index = samples[row]
                
                fnameimage= f'/testA_{index+1}.bmp'
                fnamemask= f'/testA_{index+1}_anno.bmp'
                if index >= 60:
                    fnameimage= f'/testB_{index+1-60}.bmp'
                    fnamemask= f'/testB_{index+1-60}_anno.bmp'
                    
                
                image = PIL.Image.open(f'Warwick QU Dataset (Released 2016_07_08)' + fnameimage)
                gt = PIL.Image.open(f'Warwick QU Dataset (Released 2016_07_08)' + fnamemask)
                
                with torch.no_grad():
                    result, input_image = inference_image(net, image, shouldpad=TRAIN_UNET)
                    result = postprocess(result, gt)
                
                
                with torch.no_grad():
                    result, input_image = inference_image(net, image, shouldpad=TRAIN_UNET)
                    result = postprocess(result, gt)
                if col == 0:
                    ax[row, col].imshow(image)
                elif col == 1:
                    #ax[row, col].imshow(np.array(gt) > 0)
                    ax[row, col].imshow(np.array(gt))
                else:
                    ax[row, col].imshow(image)
                    ax[row, col].imshow(result, alpha=1)
                        
                ax[row, col].set_axis_off()

            plt.savefig(f'Test_Image_Maske_Prognose{samples}.png', bbox_inches="tight")
            plt.show(); 
    
     

    def list_set(self, idx_list, listset, titles, path=''):
        '''
        Zeigt spaltenweise komplettes Bild-Masken-Set für mehrere Samples
        Eingang:
            idx_list - Liste der Indizes der darzustellenen Samples
            listset  - Liste des kompletten Bilder und Masken
            titels   - Titel der einzelnen Listenelemente
                        -> Anzahl muss mit Anzahl im listset übereinstimmen
            path - Pfad zum Abspeichern des Images-Masken-Sets.
                   Wird kein Pfad angegeben so entfällt das Speichern.
        Display:
            Bilderset bestehend aus Images und Masken                    
        '''
        rows=len(listset)
        cols=len(idx_list)
        # print(rows, cols)
        plt.figure(figsize=self.figsize)
        plt.axis("off")
        for r in range(0, rows):
            for c in range (0, cols):
                idx=idx_list[c]
                plt.subplot(rows,cols,1+r*cols + c).set_title(f'{titles[r]} {idx}', fontsize=self.fontsize)
                #print(r, idx, len(listset[r]))
                plt.imshow(listset[r][idx])
                plt.axis('off')
        plt.suptitle(f'Image-Masken-Set\n{os.path.basename(path)}\n{self.experiment}')
        if path != '':
            plt.savefig(path, bbox_inches="tight")
        plt.show()
    
    def histogramm(self, img, save_path='temp.png', abszisse='Id', ordinate='Häufigkeit H', relativ=True):
        max_id=img.max() # Maximaler Wert in der Maske
        plt.title("Histogramm\n", fontsize=16)
        plt.xlabel(abszisse, fontsize=14)
        plt.ylabel(ordinate, fontsize=14) #density=True
        n,bins,patches=plt.hist(img.ravel(), bins=int(max_id+2), range=(0, max_id+2),\
                 histtype='bar', align='left', color='gray', density=relativ, ec='k', cumulative=False )

        plt.savefig(save_path, bbox_inches="tight")                 
        #plt.show()                 
        return n,bins,patches        


   

  





            

