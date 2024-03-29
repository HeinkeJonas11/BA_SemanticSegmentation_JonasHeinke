# phyton3
# Letzte Änderung: 2021-05-27
# Autor: J. Heinke

''' Inhalt
1. SubdividePictures   - Unterteilt große Bilder in mehrere kleine Bilder
und speichert diese in separaten Verzeichnissen './subimg' und './submask'
'''
import numpy as np
from skimage import io #imread, imsave, imshow
import os
import matplotlib.pyplot as plt
from PIL import Image

class SubdividePictures(object):
    ''' 
    1. Unterteilt große Bilder in mehrere kleine Bilder
    und speichert diese in separaten Verzeichnissen './subimg' und './submask'
    '''
    def __init__(self, image_pathnames, mask_pathnames, path_sub_images, path_sub_masks,
                out_shape=(512,512), instanzen=2, verbose=True):
        '''Konstruktor '''
        super(SubdividePictures, self).__init__()
        # Quellpfade für Eingangsbilder un Masken (Targets)
        self.image_pathnames = image_pathnames
        self.mask_pathnames = mask_pathnames    
        # Größe der Aussgangsbilder
        self.out_shape=out_shape
        # Größe eines Eingangsbildes (h, w) -> wird aktualisiert
        self.in_shape=(4480, 7392)
        # Ausgabepfade für Subimages (Verzeichnisname benennt auch die Auflösung der Ausgangs-Bild-Masken)
        self.path_sub_images= str(path_sub_images) + str(out_shape)
        self.path_sub_masks = str(path_sub_masks)  + str(out_shape)
        # Erzeuge Zielpfade bzw. Zielverzeichnisse falls nicht vorhanden
        try:
            os.mkdir(self.path_sub_images)
            os.mkdir(self.path_sub_masks)
        except:
            print('Sub-Verzeichnisse existieren')
        # Objekte werden entsprechend der Instanzen unterschiedlich farblich dargestellt           
        self.instanzen=instanzen-1 # -1 -> Hintergrund wird nicht mitgezählt.
        # Ausgabesteuerung zum Testen
        self.verbose=verbose


    def allpairs(self):
        ''' Für alle Original-Bilder und Masken  '''
        for idx in range(len(self.image_pathnames)):
            self.onepair(idx)
            print(idx, end=' ') # go

    def onepair(self,idx=0):
        ''' Erzeugt aus einem Bild-Masken-Paar mehrere Subbild-Submasken-Paare
        Eingangsparameter:
            idx - Index der Verzeichnislisten für Bilder und Masken
        Speichern als Datei:
            sub_img - Liste der Subbilder  -> '/subimg/ *.png
            sub_mask - Liste der Submasken -> '/submask/ *.png
            Bemerkung: Die Dateinamen setzen sich sich aus dem Namen der
            Quelldatei und den Koordinaten der linken, oberen Ecke des
            Subbildes bezüglich des Eingangsbildes zusammen
        '''
        # https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread
        image=io.imread(self.image_pathnames[idx])
        mask=io.imread(self.mask_pathnames[idx])
        #  Eingangsbildgröße (Klassenvariable) entsprechend des Eingangsbildes aktualisieren
        # -> unterschiedliche Eingangsbildgrößen sind möglich
        self.in_shape=mask.shape #mask_shape
        # Filtert die Namen des Eingangsbildes und der zugehörigen Maske aus den Pfadbezeichnungen
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


            # Bilde Pfad mit Dateinamen eines Subbild-Masken-Paars
            sum_image_filepath=str(self.path_sub_images) + '/'+ image_filename + '_x' +str(x1)+ 'y'+str(y1)+'.png'
            sum_mask_filepath=str(self.path_sub_masks) + '/' + mask_filename + '_x'+str(x1)+ 'y'+str(y1)+'.png'
            #  Sub-Bild-Masken speichern
            io.imsave(sum_image_filepath, sub_img)
            io.imsave(sum_mask_filepath, sub_mask) #imsave() got multiple values for argument 'arr'
            # Kontrollausgabe
            if self.verbose:
                #print(sum_image_filepath)
                #print(sum_mask_filepath)
                self.pairshow(sub_img, sub_mask)
                #print('-- subimg ---')
                #print(sub_img[0:20, 0:20])
                #print('-- submask ---')
                #print(sub_mask[0:20, 0:20])


    # HILFSMETHODE
    def pairshow(self, img, mask):
        '''
        Zeigt ein Bild-Masken-Paar
        Eingangsparameter:
            img - Bild bzw. Image
            mask - zugehörige Maske
        Displayausgabe:
            img  - Bild
            mask - Maske als Falschfarbenbild    
        '''
        ## eigene Farbe für Hintergrund -> Schwarz
        #https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        viridis = cm.get_cmap('viridis', 127)
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[:40, :] = np.array([0, 0, 0, 1])
        own_cmap = ListedColormap(newcolors)
        ###----------------------------###
        #print(img.shape, mask.shape)
        plt.figure(figsize=(30,15))
        plt.subplot(221) 
        plt.imshow(img)
        #plt.colorbar()
        plt.subplot(222)
        plt.imshow(mask, rasterized=False, cmap=own_cmap, vmin=0, vmax=self.instanzen)
        plt.colorbar()


    def img_boarders(self):
        '''
        Berechnet die Punkte der oberen,linken Ecke der Sub-Bild-Masken
        eines Eingangangsbildes
        Eingangsvariablen:
            Klassen- bzw. Instanzvariablen
        Rückgabeparameter:
            points - Liste mit den Punkten der Sub-Bild-Masken            
        '''
        # Anzahl vertikal und horizontal ()
        count  = (self.in_shape[0] // self.out_shape[0] + 1,
                self.in_shape[1] // self.out_shape[1] + 1)
        # Überdeckung insgesamt, da ungerade Teilung                
        overlap= (self.out_shape[0]*count[0] - self.in_shape[0],
                self.out_shape[1]*count[1] - self.in_shape[1])
        # Überdeckung für zwei angrenzende Bilder
        offset= (overlap[0] // (count[0]-1),
                overlap[1] // (count[1]-1))
        # Punkteliste
        points=[]
        # Berechnung der Eckpunkte 
        # - letzte Spalte und letzte Zeile werden anders berechnet
        for i in range(0, count[0]-1):
            for j in range(0, count[1]-1):
                point=(i * (self.out_shape[0]- offset[0]),
                       j * (self.out_shape[1]- offset[1]))
                points.append(point)
        # rechte Randbilder
        for i in range( 0,  count[0]-1):
            point= (i * (self.out_shape[0]- offset[0]),
                    self.in_shape[1] - self.out_shape[1])
            points.append(point)    
        # unterer Randbilder
        for j in range( 0, count[1]-1):
            point=(self.in_shape[0] - self.out_shape[0],
                   j * (self.out_shape[1]- offset[1]))
            points.append(point)
        # Bild rechts, unten
        point=(self.in_shape[0] - self.out_shape[0],
               self.in_shape[1] - self.out_shape[1])
        points.append(point)
        if self.verbose:
            # Kontrollausgabe
            print (count)
            print (overlap)
            print(offset)
            print(points)
        return points            

    


   
        
	


