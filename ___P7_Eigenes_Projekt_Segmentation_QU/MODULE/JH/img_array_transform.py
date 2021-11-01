# phyton3
# Last cahne: 2021-08-02
# J. H.
# General notice: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform. ...

''' Contents
Classes with __call__ for direct access:
1. ArrayTransform - Transformation (for prediction necessary)
--------------------------------------------------------
+ Methods can be called by class name.
2. TwoClasses   - Wandelt Masken deren Objekte durchnummeriert sind (indexierte Objekte)
                  in Masken deren Hintergrund auf 0 und deren Objekte auf 1 gesetzt sind.
3. Rescale      - Skaliert neu                      
4. Scale        - Skaliert neu (Überladung zu 1.)
5. RandomCrop   - Crop randomly the image in a sample
6. ToTensor     - Convert ndarrays eines sampels in sample to Tensors
7. NpToTensor   - Konvertiert einzelnes ndarrays in einen Tensors.

Notes on the code sources:
- The classes "Rescale" and "ToTensor" have been adopted and adapted.
'''


import numpy as np
from skimage import transform #io,
import torch ## for pytorch
from torch.utils.data.sampler import SubsetRandomSampler


class ArrayTransform():
    '''
    1. Transformation (for prediction necessary)
    '''
    def __init__(self):
        super(ArrayTransform, self).__init__()

    def twoClasses(self, array):
        '''
        1.1 Converts array / masks whose objects are numbered (indexed objects) into masks
            whose background is set to 0 and whose objects are set to 1.
        Input parameter:
            array - Single mask with IDs [0, 1, 2, 3, ...]
        Return:
            array - Single mask with class numbers [0, 1]
        '''
        # For all elements of the array:
        # If element> 0 then element = 1 otherwise element = 0
        array=np.where(array > 0, 1, 0)
        return array  


class TwoClasses(object):
    ''' 2.0 Converts masks whose objects are numbered (indexed objects) into masks
            whose background is set to 0 and whose objects are set to 1.
    Input parameters (Sample):
        input  - Input array (image) remains unchanged
        target - Mask (0 - background, [1, 255] - objects) is manipulated.
    Return:
        input  - Unchanged array
        target - Array (0 - background, 1 - object)
    '''
    def __call__(self, input, target):
        # np.where(target > 0, 1, target)
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i,j]>1:
                     target[i,j]=1
        return input, target

    
class TwoClasses2(object):
    ''' 2.1 Converts masks whose objects are numbered (indexed objects) into masks
            whose background is set to 0 and whose objects are set to 1.
            - Variant 2 (for training necessary)
    Input parameters (Sample):
        input  - Input array (image) remains unchanged
        target - Mask (0 - background, [1, 255] - objects) is manipulated.
    Return:
        input  - Unchanged array
        target - Array (0 - background, 1 - object)
    '''
    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        target=np.where(target > 0, 1, 0)
        return input, target        


class Rescale(object):
    """ 3. Reskaliert die Arrays
    Input parameter (__call__):
        sample (tuple: input_, target_) - Tuple consisting of image and label (target)
    Return (__call__):
        input_, target_ - rescaled image with label """
    def __init__(self, output_size):
        ''' Constructor for instantiating a class '''
        self.output_size = output_size

    def __call__(self, input_, target_ ):
        ''' Method can be called by class name '''
        new_h, new_w = self.output_size
        input_ = transform.resize(input_, (new_h, new_w))
        target_ = transform.resize(target_, (new_h, new_w))
        return input_, target_
    

class Scale(object): #(to work)
    """ 4. Scaling the image to new size.
    Input parameter (__init__):
        outputHeight <int> - Height
        outputWidth <int> - Width
    Input parameter (__call__):
        sample (tuple image, Label) - Tuple (sample) consisting of image and label (target)
    Return (__call__):
        image, label - scaled image with label """
    def __init__(self,outputHeight=124,outputWidth=256):
        ''' Constructor '''
        # Is instance (outputHeight, outputWidth (int, tuple))
        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        image = transform.resize(image, (self.outputHeight, self.outputWidth))
        return image, label


class RandomCrop(object): #(to work)
    """ 5. Crop randomly the image in a sample.
    Input parameter:
        output_size (tuple or int) - Desired output size. If int, square crop is made.
    Input parameter (__call__):
        sample (tuple image, Label) - Tuple bestehend aus Bild und Label
    Return (__call__):
        image, label - Image, label (target)"""
    def __init__(self, output_size):
        ''' Konstruktor '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        image, label = sample[0], sample[1]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,left: left + new_w]
        return image, label

    
class ToTensor(object): #(to work)
    ''' 6. Convert ndarrays in sample to Tensors.
    Input parameter (__call__):
        sample (tuple image, Label) - Tuple consisting of image and label
    Return (__call__):
        image, label - Image and label as tensors '''
    def __call__(self, sample):
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
    """ 7. Converts single ndarrays to a tensor
    Input parameter (__call__):
        npArray (array) - ndarray of an image
    Return (__call__):
        tensor - Tensor of an image """
    def __call__(self, npArray):
        npArray=np.array(npArray, dtype=np.float32)
        npArray = npArray.transpose(2, 0, 1)
        tensor=  torch.from_numpy(npArray)
        return tensor


#############################################################################################

'''
Used experimentally in the preprocessor!
'''

class MaxPoolSample():
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
    def __init__(self, input_size=(2,2,1)):
        self.input_size=input_size
    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        input = block_reduce(input, self.input_size, np.max)

        return input 


class MaxPoolTarget():
    def __init__(self, target_size=(2,2)):
         self.target_size=target_size
       
    def __call__(self, input, target):
        # Für alle Targetelemente:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        itarget = block_reduce(target, self.target_size, np.max)

        return input 

#***** END ***********************************************************************

        
	


