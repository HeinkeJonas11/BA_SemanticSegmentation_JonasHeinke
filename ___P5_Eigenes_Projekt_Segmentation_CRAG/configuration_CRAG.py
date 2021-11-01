# phyton3
# Last Change 2021
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
    # Original images 
    h_org=1516
    w_org=1511
    # Sub-images, if any
    # Only relevant for the following program module: P5-02: Subdivide pictures from dataset
    h_sub=768 # 960, 896, 832, 768 (vierteln)
    w_sub=768
    divide=False # Training und Progonse arbeiten mit den zerteilten Bild-Masken-Paaren
    # Resize to ..., necessary otherwise "CUDA out of memory"
    h_res=768 #880 #1024 #1024 #1024 #1024 #256 #880 #768 # 512 #752 #880 #960  # orginal 1516
    w_res=768 #880 # 1024 #1024 #1024 #1024 #256 #768 # 512 #752 #880 #960 # orginal 1509
    c_res=3   # Input channels of the images (R, G, B)

    
class CfgModel():
    ''' 2. Configuration parameters of the model '''
    name_experiment='CRAG(development_2021_10_27_praezision_14)' # Key
    # Output channels of the segment classification
    c_out= 2 #2 #256
    # Model parameters
    lernrate=0.1 #0.0001 # Learning rate
    epochen=150 #600 #150 #50 #150 #600 #600   # Number of epochs
    ft=32 #start filter
    n_blocks=6 #7 #6  # Number of blocks of the U-Net
    batches=2   # Number of samples in a batch
    optimizer='SGD' #{SGD, Adam} Optimization algorithm   
    
class Postprocess():
    '''3. Post processing '''
    opening_structure=(10,10) # (2,2) (8,8) (10,10) (16,16)
    ident= 'measure'  # chain - chaincode in Python, measure - Contour code per module skimage.measure (faster)    
    
    
''' 4. Name / key of the experiment '''
EXPERIMENT=f'{CfgModel.name_experiment}_blocks{CfgModel.n_blocks}_cout{CfgModel.c_out}\
_opt{CfgModel.optimizer}_lr{CfgModel.lernrate}_ep{CfgModel.epochen}\
_h{Inputs.h_res}_w{Inputs.w_res}_ft{CfgModel.ft}'   
# STATIC
# EXPERIMENT='CRAG(open)_blocks6_cout2_optAdam_lr0.001_ep600_h1024_w1024_ft32'


class Path():
    ''' 5. Path definitions '''
    # get the project directory
    project= pathlib.Path.cwd()
    # Base path of the data set -> enter here!
    dataset = pathlib.Path('../___Datasets/CRAG_v2/')
    # Training images and masks ------------------------------
    trainimages=dataset / 'train/Images/'
    trainmasks=dataset / 'train/Annotation/'
    # Test images and masks
    testimages=dataset / 'valid/Images/'
    testmasks=dataset / 'valid/Annotation/'
    # Sub-training images and masks --------------------------
    sub_trainimages= dataset / 'sub_train_images(768, 768)'
    sub_trainmasks = dataset / 'sub_train_masks(768, 768)'
    # Sub-Test images and masks
    sub_testimages=dataset / 'sub_test_images(768, 768)'
    sub_testmasks=dataset / 'sub_test_masks(768, 768)'
    # ---------------------------------------------------
    # Trained Models Path
    model= project / 'models/'
    # Results, outputs
    results=project / 'results/'
    # -----------------------------------------------------------------------
    # The paths of the experiments are individually defined in the program.
    # ------------------------------------------------------------------------

   
    def get_filenames(self, path: pathlib.Path, dateifilter: str = '*.png', sort=False):
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

