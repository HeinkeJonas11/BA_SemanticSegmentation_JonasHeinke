# phyton3
# Last cahne: 2021-08-02
# J. H.


''' Contents
Class:
1. Prediction - Prediction of the mask
'''


import torch
import numpy as np
from MODULE.JS.transformations import normalize_01, re_normalize

class Prediction():
    ''' 1. Prediction of the mask
    '''
    def __init__(self, model, device, verbose=False):
        super(Prediction, self).__init__()
        self.model=model    #  Trained model
        self.device=device   # Device (CPU, GPU)
        self.verbose=verbose  # Output for documentation and testing

    def mask(self, img ):
        '''  Predict a mask
        Input parameters:
            img - Image for which a mask with objects (glands) is to be predicted.
        Return:
            img - Array with intensity values
            (out - Array mit Klassenwerten (0, 1, . . .))
        (Adapted for dataset CRAG-v2 - segmentation of glands)) '''
        # to Batch-Tensor       
        img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
        img = normalize_01(img)  # linear scaling to range [0-1]
        img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
        img = img.astype(np.float32)  # typecasting to float32
        # Prognose
        self.model.eval()
        x = torch.from_numpy(img).to(self.device)  # to torch, send to device
        with torch.no_grad():
            y_ = self.model(x)  # send through model/network
        # to Mask-Image, Mask-Array
        out = torch.softmax(y_, dim=1)  # perform softmax on outputs
        out = torch.argmax(out, dim=1)  # perform argmax to generate 1 channel
        out = out.cpu().numpy()  # send to cpu and transform to numpy.ndarray
        out = np.squeeze(out)  # remove batch dim and channel dim -> [H, W]
        img = re_normalize(out)  # scale it to the range
        return img    