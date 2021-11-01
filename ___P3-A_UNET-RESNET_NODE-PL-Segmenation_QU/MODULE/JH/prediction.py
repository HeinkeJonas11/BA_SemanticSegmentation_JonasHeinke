import torch
import numpy as np
from MODULE.JS.transformations import normalize_01, re_normalize

class Prediction():
    ''' 1. Prognose

    '''

        #Konstruktor
    def __init__(self, model, device, verbose=False):
        super(Prediction, self).__init__()
        self.model=model    #  Trainiertes Modell
        self.device=device   # Gerät ( CPU, GPU)
        self.verbose=verbose  #Ausgabe zwecks Dokumentation und zum Testen

    def mask(self, img ):
        '''  Prognose eine Maske
        Eingang:
            img - Bild für das eine Maske mit Objekten (Drüsen) zu prognostizieren ist
        Rückgabe:
            img - Array mit Intensitätswerten bis 255
            out - Array mit Intensitätswerten (1, 2, ...)
        (Angepasst für Framework und Dataset CRAG-v2 - Segmentation von Drüsen) '''
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
        img = re_normalize(out)  # scale it to the range [0-255]

        return img    