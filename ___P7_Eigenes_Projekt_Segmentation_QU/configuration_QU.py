# phyton3
# Letzte Änderung: 2021
# Jonas Heinke

''' Contents
Classes, methods, constants for the configuration and parameterization of the programs
InpImg      - 1. Definition of the input images for the network
CfgMod      - 2. Configuration parameters of the model
Postprocess - 3. Postprocessing (morphological operation)
EXPERIMENT  - 4. Constant denotes the experiment
Path        - 5. Path information
get_filenames - 6. Method returns a list of the files including the paths of a directory
'''

import pathlib
import os
import glob


class Inputs():
    ''' 1. Definition of the input images and masks'''
    # Originalbilder
    h_org=522
    w_org=775
    # Resize to ...
    h_res=352 #512 #352 
    w_res=512 #768 # 512 
    c_res=3   # Input channels of the images (R, G, B)

    
class CfgModel():
    ''' 2. Configuration parameters of the model '''
    name_experiment='QU_vgl(open8x8_2021-08-26)' # Key
    # Output channels of the segment classification
    c_out= 2 #2 #256
    # Model parameters
    lernrate=0.001 #0.0001 # Lernrate, Standard 0.01
    epochen=600 #600 #150 #50 #150 #600 #600   # Anzahl der Epochen
    ft=64 #start filter
    n_blocks=5  # # Number of blocks of the U-Net (Auflösungsabhängig, max=6)
    batches=2   # Number of samples in a batch
    optimizer='Adam' #{SGD, Adam} Optimization algorithm


class Postprocess():
    '''3. Post processing '''
    opening_structure=(8, 8)
    ident= 'measure'  # chain - chaincode in Python, measure - Contour code per module skimage.measure (faster)


''' 4. Name / Key of the experiment '''
EXPERIMENT=f'{CfgModel.name_experiment}_blocks{CfgModel.n_blocks}_cout{CfgModel.c_out}\
_opt{CfgModel.optimizer}_lr{CfgModel.lernrate}_ep{CfgModel.epochen}\
_h{Inputs.h_res}_w{Inputs.w_res}_ft{CfgModel.ft}' 
# STATIC
# EXPERIMENT='QU(open8x8)_blocks5_cout2_optAdam_lr0.001_ep600_h352_w512_ft32'


class Path():
    ''' 5. Path definitions '''
    # Liefert das Projektverzeichnis
    project= pathlib.Path.cwd()
    # Basispfad des Datensets
    dataset = project / 'Warwick QU Dataset (Released 2016_07_08)/'
    # Trainingsbilder und Masken
    trainimages=dataset 
    trainmasks=dataset 
    # Testbilder und Masken
    testimages=dataset 
    testmasks=dataset 
    # Pfad für trainierte Modelle
    model= project / 'models/'
    # Ergebnisse, Ausgaben, Protokolle
    ## QU stellt zwei Test-Serien bereit (A, B).
    serie='A' # [A, B]
    results=project / 'results/'
    if serie=='B':
        results=project / 'results_B/'

    def get_filenames(self, path: pathlib.Path, dateifilter: str = '*.bmp', sort=False):
        ''' 6. Help method
        Provides a list of files in a directory, independent of the system.
        Entrance:
             path - path in which the files are searched for.
             file filter - filter for selecting files, default: * .png
             sort - Determines the order of the list elements [random, string-sorted]
         Return:
             list_of_filenames - list of filenames
        '''
        list_of_filenames = [file for file in path.glob(dateifilter) if file.is_file()]
        if sort:
            list_of_filenames.sort()
        return list_of_filenames








        
	


