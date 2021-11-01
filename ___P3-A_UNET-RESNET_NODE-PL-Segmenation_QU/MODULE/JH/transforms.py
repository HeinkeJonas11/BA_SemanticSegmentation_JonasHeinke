# phyton3
# Letzte Änderung: 2021-06-02
# Allgemeiner Hinweis: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform. ...

''' Inhalt
Klassen mit __call__ zum Direktaufruf:
1. TwoClasses   - Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
                  in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
2. Rescale      - Skaliert neu                      
3. Scale        - Skaliert neu (Überladung zu 1.)
4. RandomCrop   - Crop randomly the image in a sample
5. ToTensor     - Convert ndarrays eines sampels in sample to Tensors
6. NpToTensor   - Konvertiert einzelnes ndarrays in einen Tensors.

Hinwis zu den Codequellen:
- Die Klassen 2 bis 5 wurden übernommen und angepasst.

'''


import numpy as np
from skimage import transform #io,
from skimage.measure import block_reduce
import torch ## for pytorch
from torch.utils.data.sampler import SubsetRandomSampler



class ImageTransform():
    '''  IN WORK
    '''
        #Konstruktor
    def __init__(self):
        super(ImageTransform, self).__init__()

    def twoClasses(self, array):
        # Für alle Elemente des Arrays:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        array=np.where(array > 0, 1, 0)
        return array  


#======== Direkt aufrufbare Klassen über __call__ ====================================

class TwoClasses(object):
    ''' 1.0 Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
    in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
    Eingang (Sample):
        input - Eingangsarray (Bild) bleibt unverändert
        target - Maske (0 - Hintergrund, [1, 255] - Objekte) wird manipuliert
    Return:
        input - Unverändertes Array
        target - Array (0 - Hintergrund, 1 - Objekt)
    '''
    def __call__(self, input, target):
        # np.where(target > 0, 1, target)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i,j]>1:
                     target[i,j]=1
        return input, target

class TwoClasses2(object):
    ''' 1.1 Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
    in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
    Eingang (Sample):
        input - Eingangsarray (Bild) bleibt unverändert
        target - Maske (0 - Hintergrund, [1, 255] - Objekte) wird manipuliert
    Return:
        input - Unverändertes Array
        target - Array (0 - Hintergrund, 1 - Objekt)
    '''
    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        target=np.where(target > 0, 1, 0)
        return input, target     


#****** Versuchsweise im Preprozessor verwendet 
class MaxPoolSample():
    ''' 1.1 Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
    in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
    Eingang (Sample):
        input - Eingangsarray (Bild) bleibt unverändert
        target - Maske (0 - Hintergrund, [1, 255] - Objekte) wird manipuliert
    Return:
        input - Unverändertes Array
        target - Array (0 - Hintergrund, 1 - Objekt)
    '''
    def __init__(self, input_size=(2,2,1), target_size=(2,2)):
        self.input_size=input_size
        self.target_size=target_size

    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        input = block_reduce(input, self.input_size, np.max)
        target = block_reduce(target, self.target_size, np.max)
        return input, target



class MaxPoolInput():
    ''' 1.1 Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
    in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
    Eingang (Sample):
        input - Eingangsarray (Bild) bleibt unverändert
        target - Maske (0 - Hintergrund, [1, 255] - Objekte) wird manipuliert
    Return:
        input - Unverändertes Array
        target - Array (0 - Hintergrund, 1 - Objekt)
    '''
    def __init__(self, input_size=(2,2,1)):
        self.input_size=input_size
       

    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        input = block_reduce(input, self.input_size, np.max)

        return input 


class MaxPoolTarget():
    ''' 1.1 Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
    in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
    Eingang (Sample):
        input - Eingangsarray (Bild) bleibt unverändert
        target - Maske (0 - Hintergrund, [1, 255] - Objekte) wird manipuliert
    Return:
        input - Unverändertes Array
        target - Array (0 - Hintergrund, 1 - Objekt)
    '''
    def __init__(self, target_size=(2,2)):
         self.target_size=target_size
       
    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        itarget = block_reduce(target, self.target_size, np.max)

        return input 

#*****
# -----------------------------------------------------
class Rescale(object):
    """ 2. Reskaliert die Arrays
    Parameter (__call__):
        sample (tuple: input_, target_) - Tuple bestehend aus Bild und Label
    Return (__call__):
        input_, target_ - reskaliertes Bild mit Label """
    def __init__(self, output_size):
        ''' Konstruktor zur Instanzierung einer Klasse '''
        self.output_size = output_size

    def __call__(self, input_, target_ ):
        ''' Methode  per Klassenname aufrufbar '''
   
        new_h, new_w = self.output_size
        input_ = transform.resize(input_, (new_h, new_w))
        target_ = transform.resize(target_, (new_h, new_w))
        return input_, target_
    

class Scale(object): #(to work)
    """ 3. Scalieren des Bildes auf neu Größe
    Parameter (__init__):
        outputHeight <int> - Ausgabehöhe
        outputWidth <int> - Ausgabeweite
    Parameter (__call__):
        sample (tuple image, Label) - Tuple bestehend aus Bild und Label
    Return (__call__):
        image, label - skaliertes Bild mit Label """
    def __init__(self,outputHeight=124,outputWidth=256):
        ''' Konstruktor '''
        #assert isinstance(outputHeight, outputWidth, (int, tuple))
        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
    def __call__(self, sample):
        ''' Methode  per Klassenname aufrufbar '''
        image, label = sample[0], sample[1]
        image = transform.resize(image, (self.outputHeight, self.outputWidth))
        return image, label


class RandomCrop(object): #(to work)
    """ 4. Crop randomly the image in a sample.
    Parameter:
        output_size (tuple or int) - Desired output size. If int, square crop is made.
    Parameter (__call__):
        sample (tuple image, Label) - Tuple bestehend aus Bild und Label
    Return (__call__):
        image, label - Bild mit Label """
    def __init__(self, output_size):
        ''' Konstruktor '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        ''' Methode  per Klassenname aufrufbar '''
        image, label = sample[0], sample[1]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,left: left + new_w]
        return image, label

class ToTensor(object): #(to work)
    ''' 5. Convert ndarrays in sample to Tensors.
    Parameter (__call__):
        sample (tuple image, Label) - Tuple bestehend aus Bild und Label
    Return (__call__):
        image, label - Bild und Label als tensoren '''
    def __call__(self, sample):
        ''' Methode  per Klassenname aufrufbar
        Parameter:
            sample - (x-Feld, y-Label)'''
        image, label = sample[0], sample[1]
        image=np.array(image, dtype=np.float32)   # als typ (Label) werden nur Int und Float axzeptiert
        label=np.array(label, dtype=np.long)   # als typ (Label) werden nur Int und Float axzeptiert
        # swap color axis because # (ggf. nicht notwendig, je nach Modell)
        # numpy image: H x W x C ->   # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image= torch.from_numpy(image)
        label= torch.from_numpy(label)
        return image, label


class NpToTensor(object): #(to work)
    """ 6. Konvertiert einzelnes ndarrays in einen Tensors
    Parameter (__call__):
        npArray (array) - ndarray (Bild) 
    Return (__call__):
        tensor - Tensor (Bild) """
    def __call__(self, npArray):
        ''' Methode  per Klassenname aufrufbar '''
        npArray=np.array(npArray, dtype=np.float32)
        npArray = npArray.transpose(2, 0, 1)
        tensor=  torch.from_numpy(npArray)
        return tensor




        
	


