# phyton3
# Letzte Änderung: 2021-10-27

''' Content
-------------------------------------------------------------------------------
Prediction  - 1. Prediction of a mask
------------------------------------------------------------------------------
'''

import torch
import numpy as np
from MODULE.JS.transformations import normalize_01, re_normalize


class Prediction():
    '''
    1. Prediction of a mask
    '''
    #Konstruktor
    def __init__(self, model, device, verbose=False):
        super(Prediction, self).__init__()
        self.model=model    #  Trainiertes Modell
        self.device=device   # Gerät ( CPU, GPU)
        self.verbose=verbose  #Ausgabe zwecks Dokumentation und zum Testen

    def mask(self, img ):
        '''  Prediction of a mask
        (Adapted for framework and dataset CRAG-v2 - segmentation of glands)
        Input parameters:
            img - Image for which a mask with objects (glands) is to be predicted.
        Rückgabe:
            img - Array with intensity values up to 255.
            out - Array with intensity values (1, 2, ...)
         '''
        # to Batch-Tensor       
        img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
        img = normalize_01(img)  # linear scaling to range [0-1]
        img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
        img = img.astype(np.float32)  # typecasting to float32
        # Prediction
        self.model.eval()
        x = torch.from_numpy(img).to(self.device)  # to torch, send to device
        with torch.no_grad():
            y_ = self.model(x)  # send through model/network
        # to Mask-Image, Mask-Array
        out = torch.softmax(y_, dim=1)  # perform softmax on outputs
        out = torch.argmax(out, dim=1)  # perform argmax to generate 1 channel
        out = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray
        out = np.squeeze(out)  # remove batch dim and channel dim -> [H, W]
        img = re_normalize(out)  # scale it to the range [0-255]
        return img    