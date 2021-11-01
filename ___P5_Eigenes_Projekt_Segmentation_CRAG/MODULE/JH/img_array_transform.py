# phyton3
# Letzte Änderung: 2021-06-02
# Allgemeiner Hinweis: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform. ...

''' Content: Classes with __call__ for direct access:
(Classes 2, 4, 5 have been adopted and adapted.)
--------------------------------------------------------------------------------
TwoClasses,     - 1. Converts masks whose objects are numbered (indexed objects)
TwoClasses2       into masks whose background is set to 0 and
                    whose objects are set to 1.
--------------------------------------------------------------------------------                    
2. Rescale      - 2. Rescaled the arrays.
--------------------------------------------------------------------------------
3. Scale        - 3. Scaling the image to new size.
--------------------------------------------------------------------------------
4. RandomCrop   - 4. Crop randomly the image in a sample
--------------------------------------------------------------------------------
5. ToTensor     - 5. Convert ndarrays eines sampels in sample to Tensors
--------------------------------------------------------------------------------
6. NpToTensor   - 6. Konvertiert einzelnes ndarrays in einen Tensors.
-------------------------------------------------------------------------------
'''


import numpy as np
from skimage import transform #io,
import torch ## for pytorch
from torch.utils.data.sampler import SubsetRandomSampler


class ArrayTransform():
        #Konstruktor
    def __init__(self):
        super(ArrayTransform, self).__init__()

    def twoClasses(self, array):
        # Für alle Elemente des Arrays:
        # Wenn Element > 0 dann Element=1 ansonsten Element=0
        array=np.where(array > 0, 1, 0)
        return array  


#======== Directly callable classes via __call__ ====================================

class TwoClasses(object):
    ''' 1.0 Converts masks whose objects are numbered (indexed objects)
    into masks whose background is set to 0 and whose objects are set to 1.
    Eingang (Sample):
        input - Input array (image) remains unchanged.
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
    ''' Converts masks whose objects are numbered (indexed objects)
    into masks whose background is set to 0 and whose objects are set to 1.
    Input (Sample):
        input - Input array (image) remains unchanged.
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

# -----------------------------------------------------
class Rescale(object):
    """ 2. Reskaliert die Arrays
    Parameter (__call__):
        sample (tuple: input_, target_) - Tuple consisting of image and label
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
    """ 3. Scaling the image to new size.
    Parameter (__init__):
        outputHeight <int> - Output height
        outputWidth <int> - Output width
    Parameter (__call__):
        sample (tuple image, Label) - Tuple consisting of image and label
    Return (__call__):
        image, label - scaled image with label """
    def __init__(self,outputHeight=124,outputWidth=256):
        ''' Constructor for instantiating a class '''
        #assert isinstance(outputHeight, outputWidth, (int, tuple))
        self.outputHeight = outputHeight
        self.outputWidth = outputWidth
    def __call__(self, sample):
        ''' Method can be called by class name '''
        image, label = sample[0], sample[1]
        image = transform.resize(image, (self.outputHeight, self.outputWidth))
        return image, label


class RandomCrop(object): #(to work)
    """ 4. Crop randomly the image in a sample.
    Parameter:
        output_size (tuple or int) - Desired output size. If int, square crop is made.
    Parameter (__call__):
        sample (tuple image, Label) - Tuple consisting of image and label.
    Return (__call__):
        image, label - Image with label """
    def __init__(self, output_size):
        ''' Construktor '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, sample):
        ''' Method can be called by class name  '''
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
        sample (tuple image, Label) - Tuple consisting of image and label.
    Return (__call__):
        image, label - Bild und Label als tensoren '''
    def __call__(self, sample):
        ''' Method can be called by class name.
        Parameter:
            sample - (x-array, y-label)'''
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
        npArray (array) - ndarray (Image) 
    Return (__call__):
        tensor - Tensor (Image) """
    def __call__(self, npArray):
        '''Method can be called by class name. '''
        npArray=np.array(npArray, dtype=np.float32)
        npArray = npArray.transpose(2, 0, 1)
        tensor=  torch.from_numpy(npArray)
        return tensor




        
	


