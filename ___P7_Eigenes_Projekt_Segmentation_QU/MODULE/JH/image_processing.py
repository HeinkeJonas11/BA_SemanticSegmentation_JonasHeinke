# image processing
# phyton3
# Letzte Änderung: 2021-09-08
# J.H

'''
Inhalt
Klassen:
IdentifyObject - 1. Image and array operations
                    - Chain code of the objects in a mask
                    - Filling holes
# ------------------
DrawInArray() -  2. Zeichnet in ein Array, zum Beispiel eine Kontur

    Methoden:
    a_chaincode - 2.1 Verwendet den Kettencode um eine Kontur in eine Array zu übertragen.
#-----------------------------------------
ObjectIDsOfArray() - 3. Second variant for identifying individual objects within an array.
                        Detects contiguous areas within an array.
https://scikit-image.org/docs/dev/api/skimage.measure.html (skimage.measure.label)

#----------------------------------------------------------------------
DrawChainInArray() - 4. Draws the chain code in an array, here a contour.
# ------------------------------------    
Convert()  -  5. Konvertiert Arrays und Images
    Methode:
    chains_to_contourcodes_2 - 6.2 Converts a chain code to a contour code
# -----------------------------------------------                             
'''

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import scipy.ndimage
from scipy import ndimage
import skimage.measure # für zweite Version zur Objektidentifikation

''' Auch in dem Modul image_prozessing'''
class IdentifyObject():
    ''' 
    1. Image and array operations
    '''
    #Konstruktor
    def __init__(self, inputarray, verbose=False):
        super(IdentifyObject, self).__init__()
        # Eingangsbild als zweidimensionales Nupy-Array
        self.array = inputarray
        # Fügt einen Rand mit 0-Werten hinzu. Rand ist notwendig, damit Objekte umrundet werden können.   
        self.array = self.padding_array(self.array)
        # Zur Ausgabe zwecks Dokumentation und zum Testen
        self.verbose = verbose  
        # Enthält alle bereits gefundenen Objekte.
        #  Anfangswerte sind 0, wie Hintergrund.
        self.chain_array = np.zeros(self.array.shape).astype(np.int16)

    def chaincode(self, class_id, id_=1):
        ''' 1.1 Pre- and post-processing to determine the chain code
                including method call for calculation. 
        Class variable input:
            self.array - Array with objects
        Input parameters:
            class_id - Related elements with the intensity value are regarded as an object.
                (It is a value != 0, usually == 1)
            id_       - new value for element
        Return parameters:
            chaincodes - List of chain codes
                Example of a list entry: 
                chaincodes[0]=[[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
            self.chain_array - Array of similar objects. Each object has a unique ID.
                                All elements (pixels) of an object (gland) have the same ID.
            self.array - Array with classes
        '''
        ## Hintergrund auf -1 setzen, um Hintergrund von Löchern in Objekten zu unterscheiden.
        self.array = self.fill_ground(self.array, ground=-1)
        # print(array_1)
        # Füllt Löcher der Objekte - IN WORK
        self.array = self.fill_objects(self.array, class_id)
        # Liefert den Kettencode für Kontur der Objekte. Äußere Umrandung muss 0 oder -1 sein.
        # Vergibt jedem Objekt eine eindeutige Identifikation (Id).Füllt Löcher in den Objekten.
        chaincodes, self.chain_array = self.chaincode_run(class_id, id_)
        # Alle Hintergrundelemente von -1 auf 0 setzen (0 ist Standardwert für Hintergrund)
        self.chain_array = self.fill_ground_zero(self.chain_array )
        # Rand entfernen. Rand war notwendig, damit Objekte umrundet werden können.
        self.chain_array = self.unpadding_array(self.chain_array)
        # Startpunkt des Kettencods verschieben, damit Kettencode für Array ohne Rand gültig.
        chaincodes=self.unpadding_chaincode(chaincodes)
        print(end='|')
        return chaincodes, self.chain_array, self.array
   
    def chaincode_run(self, class_id, id_): 
        '''
        1.2 Returns the chain code of all objects in an array.
        Each object is given a unique identification number.
        In addition, an array is created with the identified objects.
        A starting point or a starting point is searched for for these objects.
        From there the border of the object begins and ends.
        The outlined areas are also filled in (see return parameters).
        Note on the indices of the arrays: array [i, j] i - rows, j - columns 
        Class variable input:
            self.array - Array with contiguous areas
        Input parameters:
            id_      - Array-Index
            class_id - Related elements with the intensity value are regarded as an object.
        Return parameters:
            chaincodes - List of chain codes
            self.chain_array - Array of similar objects.
                                Each object has a unique ID.
                                All elements of an object have the same ID.
        '''
        start_chain = np.array([0,0])
        neighbor = np.array([0,0])
        point_of_chain=np.array([0,0])
        chaincodes = []
        gefunden = False
        while True:
            for i in range(start_chain[0], self.array.shape[0]):
                if gefunden: # So that both loops can be left.
                    break
                for j in range(0, self.array.shape[1]):
                    # SEARCH THE STARTING POINT OF AN OBJECT.
                    # Find an object with intensity class_id that has not yet been found.
                    if self.array[i,j] == class_id and (self.chain_array[i, j]<=0):
                        # Starting point found
                        chain=[]
                        start_chain=[i, j]
                        chain.append(start_chain)
                        self.chain_array[i,j]=id_
                        gefunden=True
                        break
                        # --- Starting point of a new object found ---
            # FOLLOW THE CONTOUR NOW
            # Neighbor indices as a shift in i, j
            shift = np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])
            # The contour tracing begins with the neighbor 4.
            # It is not an object point because it is to the left of the contour found.
            point_of_chain = start_chain
            s = 0
            s_check = s
            zyklus = 0
            while True:
                # We are looking for the intensity transition from object_id (== 1)
                # to not object_id (<= 0)
                while True:
                    s += 1
                    zyklus += 1
                    # s_check grows continuously by 1, but must be reduced to the interval [0, 8[
                    s_check = s % 8
                    neighbor= point_of_chain + shift[s_check]
                    if self.verbose: #----------------------------------------------------------
                        print('vor : ',s ,s_check, neighbor, point_of_chain,\
                              start_chain,'neighborwert = ',self.array[neighbor[0], neighbor[1]])
                    if self.array[neighbor[0], neighbor[1]] == class_id or zyklus >8:
                        zyklus=0
                        break
                ''' while_end to find an edge point'''        
                # Edge, contour point found *
                try:
                    chain.append(s_check)
                except:
                    print(f'No starting point? No object found in array [{id_}]!')
                    
                self.chain_array[neighbor[0], neighbor[1]] = np.uint8(id_)
                # take a step or two back depending on which neighbors you find
                s -= 3
                if neighbor[0] !=  point_of_chain[0] and neighbor[1] != point_of_chain[1]:
                    s -= 1
                    if self.verbose:
                        print('diagonal connection!)')
                else:
                    if self.verbose:
                        print('straight connection!')                    
                point_of_chain=neighbor
                if point_of_chain[0] == start_chain[0] and point_of_chain[1] == start_chain[1]:
                    break
                ''' while_end for outlining a contour with the id_'''
            ##! Filling of the object area with the id_ including any holes still
            ##  present in the object areas.
            ##! Necessary so that the starting point of the next object can be found.
            self.chain_array=self.fill_object(self.chain_array, np.uint8(id_) )
            # Add the chain code to the list
            try:
                chaincodes.append(chain)
            except:
                print('No chain code to append!')
            else:
                pass                
            # Prepare search for next contour
            id_ += 1
            s = s % 8 # Avoid danger
            gefunden=False
            # searched everything
            if j >= self.array.shape[1] - 1 and i >= self.array.shape[0] - 1:
                break
            ''' while_end: To find all objects of an image for a given intensity value '''    
        return chaincodes, self.chain_array

    def padding_array(self, array):
        '''
        1.3 Adds a border to an array.
        Input parameters:
            array - Two-dimensional array
        Return parameters:
            array_padding - Array with an additional margin
        '''
        array_padding=np.zeros((array.shape[0]+2, array.shape[1]+2))
        array_padding[1:-1, 1:-1]=array[:, :]
        return array_padding

    def unpadding_array(self, array_padding):
        '''
        1.4 Removes margin rows and margin columns from an array.
        Input parameters:
            array_padding - Two-dimensional array whose margin is to be removed.
        Return parameters:
            array - Array now without a margin.
        '''
        array=np.zeros((array_padding.shape[0]-2, array_padding.shape[1]-2))
        # print(self.array.shape, array.shape)
        array[:,:] = array_padding[1:-1, 1:-1]
        return array

    def unpadding_chaincode(self, chaincodes):
        '''
        1.5 Moves the starting point of the chain code one line up and one column to the left.
            This is necessary because the chain code was determined for objects in the array
            with a margin.
        Input parameters:
            chaincodes - List of chain codes
            Example of a list entry: [[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
        Return parameters:
            chaincodes - List of corrected chain codes
        '''
        for id_ in range(len(chaincodes)):
                chaincodes[id_][0][0] -= 1
                chaincodes[id_][0][1] -= 1
        return chaincodes

    def fill_object(self,  chain_array, id_=1):
        '''
        1.6 Fills outlined objects of the chain_array with the specified id.
            (Simple but not perfect algorithm)
        Input parameters:
            id_ - Identification number of the object, object is already outlined.
                  The object area is filled with this ID.
            chain_array - Array with n objects. Each object has its own ID, whose margin is marked.
        Return parameters:
            chain_array - Array with n objects. The inner area is filled with the associated ID.
        '''
        for i in range(chain_array.shape[0]):
            for j in range(chain_array.shape[1]):
                # ist links, rechts, oben und unten eine Grenze des Objektes?
                up   = len(np.where(chain_array[0: i, j] == id_)[-1])
                down = len(np.where(chain_array[i: , j] == id_)[-1])
                left = len(np.where(chain_array[i,  0:j] == id_)[-1])
                right= len(np.where(chain_array[i, j: ] == id_)[-1] )
                if (up > 0 and down > 0 and left > 0 and right > 0 and chain_array[i, j] == 0):
                    chain_array[i,j] = id_
        return chain_array

    def fill_objects(self, chain_array, id_):
        '''
        1.7 Changes object elements with the value = 0 of the chain_array, sets the specified id_
        Input parameters:
            chain_array - Array with n objects.
            id_ - new ID                       
        Return parameters:
             chain_array - Array with n objects. The area is filled with the associated ID.
        '''
        for i in range(chain_array.shape[0]):
            for j in range(chain_array.shape[1]):
                if chain_array[i,j] == 0: # There is a hole in the object.
                    chain_array[i,j]=id_
        return chain_array

    def fill_ground(self, chain_array, ground = -1):
        '''
        1.8 Fills the background of the chain_array with ground values.
        - Background elements are connected.
        - Holes are inside objects and remain unchanged.
        Input parameters:
            chain_array - Array with n objects. Each object has its own ID,
                            the border of which is marked.
            ground - Background value                         
        Return parameters:
                chain_array - Array with n objects. The background is completely filled with -1.
                                Inner surfaces of objects retain the value 0.     
        '''
        # Set border on background (border was added previously)
        for i in range(chain_array.shape[0]):
            chain_array[i, 0] = ground
            chain_array[i, chain_array.shape[1]-1] = ground
        for j in range(chain_array.shape[1]):
            chain_array[0, j]=ground
            chain_array[chain_array.shape[0]-1, j] = ground
        
        ### Neighbors, relative
        # --------------------------------------------------------------------------
        shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])
        def fill(): ### LOCAL METHOD
            ''' 1.9 Fills neighboring elements if they belong to the background.
            Return parameters:
                chain_array - 
            '''
            for n in range(len(shift)):
                nachbar=chain_array[i+shift[n, 0], j+shift[n,1]]  
                if nachbar == ground and chain_array[i, j] == 0:
                    chain_array[i, j] = ground
                    break
        #----------------------------------------------------
        # Forward run, line by line
        for i in range(1,chain_array.shape[0] - 1):
            for j in range(1,chain_array.shape[1] - 1):
                fill()
        #-> Several directions (loops) have to be processed to fill the shadows.
        for j in range(1,chain_array.shape[1] - 1):
            for i in range(1,chain_array.shape[0] - 1):
                fill()
        for i in reversed(range(1,chain_array.shape[0] - 1)):
            for j in reversed(range(1,chain_array.shape[1] - 1)):
                fill()
        for j in reversed(range(1,chain_array.shape[1] - 1)):
            for i in range(1,chain_array.shape[0] - 1):
                fill()
        return chain_array

    def fill_ground_zero(self, chain_array, ground = -1):
        ''' 1.10 Set the background pixels from -1 to 0
        Input parameters, Return parameters:
            chain_array - Array with 0 for background'''
        for i in range(chain_array.shape[0]):
            for j in range(chain_array.shape[1]):
                if chain_array[i,j] == ground:
                    chain_array[i,j] = 0
        return chain_array  


class DrawInArray():
    ''' 2. Draws in an array, here a contour.
        Note: Faster calculation if class in the notebook.'''
    #Konstruktor
    def __init__(self, array, verbose):
        super(DrawInArray, self).__init__()
        # Eingangsbild als zweidimensionales Numpy-Array
        self.array = array
        # Steurt Ausgabe zur Kontrolle    
        self.verbose = verbose
        # Definition des Kettencods, Nachbarindex und Verschiebung in i,j
        self.shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])

    def a_chaincode(self, chaincode, element_value=255):
        '''
        2.1 Verwendet den Kettencode um eine Kontur in eine Array zu übertragen.
        Class variables:
            self.array - Array, dass verändert wird
        Input parameters:
            chaincode - Kettencode entsprechend der Definition in self.shift
            element_value - Wert, den die Konturpixel erhalten
        Return parameters:
            self.array - Array mit Kontur entsprechend des Kettencods
        '''
        pi = int(chaincode[0][0])
        pj = int(chaincode[0][1])
        self.array[pi, pj] = element_value # Startpunktvalue
        if self.verbose:
            print(f'\na_chaincode -> Starting point: {pi}, {pj}')
        for cod in range(1, len(chaincode)):
            i = int(self.shift[chaincode[cod]][0])
            j = int(self.shift[chaincode[cod]][1])
            pi += i
            pj += j
            self.array[pi, pj]=element_value #element_value
            if self.verbose:
                print(f'P: {pi}, {pj}',end=','  )
        return self.array 
    
    def get_array(self):
        return self.array


class ObjectIDsOfArray():    
    ''' 3. Second variant for identifying individual objects within an array.
        Detects contiguous areas within an array.
        https://scikit-image.org/docs/dev/api/skimage.measure.html (skimage.measure.label)
        
    '''
    def __init__(self):
        '''Constructor'''
        super(ObjectIDsOfArray, self).__init__()
    def getObjects(self, mask, background=0):
        ''' 3.1 Identifies related areas and gives them an Id.
        Input parameters:
            mask - Array for mask
            background - Background identity
        Return parameters:
            contur_codes - Array with coordinates of the contours
            object_array - Array with contours
        '''
        contur_codes = skimage.measure.find_contours(mask, background)
        # Combines contiguous areas into one object
        object_array = skimage.measure.label(mask)
        return contur_codes, object_array

    def fill_objects(self, mask, structure):
        ''' 3.2 Morphological operations to post-process the predicted masks.
        - Variant for identifying individual objects based on the module "inteference_utily.py"
        Input parameters:
            mask - Array for mask
            structure - Filter, kernel for morphological operation
        Return parameters:
            mask_morph - Morphologically processed mask
        '''
        # Entfernt kleinste Objekte in Größe des Strukturelements
        mask_morph = ndimage.binary_opening(mask, structure = np.ones(structure)).astype(int)
        for i in np.unique(mask_morph): # For all occurring intensities (here only 0, 1)
            if i == 0: continue # If hole or background
            # Fills holes within the objects / glands
            filled = scipy.ndimage.morphology.binary_fill_holes(mask_morph == i)
            mask_morph[mask_morph == i] = 0 # background
            mask_morph[filled == 1] = i     # Hole gets value i
        return mask_morph        


class DrawChainInArray():
    ''' 4. Draws the chain code in an array, here a contour.'''
    def __init__(self, array, verbose):
        super(DrawChainInArray, self).__init__()
        self.array = array # Input array as a two-dimensional numpy array
        self.verbose = verbose # Controls output
        # Definition of the chain code, neighbor indices and shift in i, j
        self.shift = np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])

    def a_chaincode(self, chaincode, element_value=255, depth=1):
        '''
        4.1 Uses the chain code to transfer a contour into an array.
        Class variables:
            self.array - Array that is changed
        Input parameters:
            chaincode - Chain code as defined in self.shift
            element_value -Value that the contour points receive
            depth - Thickness of the contour train (contour thickness)
        Return parameters:
            self.array - Array with contour according to the chain code
        '''
        pi = int(chaincode[0][0])
        pj = int(chaincode[0][1])
        self.array[pi, pj]=element_value # Startpunktvalue
        if self.verbose:
            print(f'\na_chaincode -> Startpunkt: {pi},{pj}')
        for cod in range(1, len(chaincode)):
            i = int(self.shift[chaincode[cod]][0])
            j = int(self.shift[chaincode[cod]][1])
            pi += i
            pj += j
            try:
                self.array[pi, pj]=element_value #element_value
                for d in range(0, depth): # Umrandungsstärke
                    self.array[pi + d, pj] = element_value #element_valu
                    self.array[pi - d, pj] = element_value #element_valu
                    self.array[pi, pj + d] = element_value #element_valu
                    self.array[pi, pj - d] = element_value #element_valu
            except:
                pass
            if self.verbose:
                print(f'P: {pi}, {pj}',end = ','  )
        return self.array 
    
    def get_array(self):
        ''' 4.2 Getter for class variable
        Return parameters:
            self.array - Array with contour according to the chain code'''
        return self.array
# ----------------------------------------------------------------------
class DrawCodeInArray():
    ''' 5. Draws a contour in an array.'''
    def __init__(self, verbose):
        super(DrawCodeInArray, self).__init__()
        self.verbose = verbose # Controls output

    def all_contours(self, contour_array, contour_codes, element_value=255, depth=1):
        '''
        5.1 Uses the contour code to transfer a contour into an array.
        Input parameters:
            contour_array - Array that is changed
            contour_codes - Coordinates of the contours (several per array)
            element_value - Value that the contour points receive.
            depth - Thickness of the contour (contour thickness)
        Return parameters:
            contour_array- Array with contour corresponding to the contour code.
        '''
        for contour in contour_codes:
            for point in contour:
                contour_array[int(point[0]), int(point[1])] = element_value
                try:
                    for d in range(0, depth): # Kontour dicker zeichnen
                        contour_array[int(point[0] + d), int(point[1]) + d] = element_value
                        contour_array[int(point[0] - d), int(point[1]) - d] = element_value
                except:
                    pass
        return contour_array
                    

class Convert():
    '''  
    6. Converts arrays and images
    '''
    def __init__(self):
        super(Convert, self).__init__()

    def to_3d_array(self, array2d, dtype=int):
        ''' 6.1         # # IN WORK # #
        '''
        elemente=array2d.shape[0] * array2d.shape[1] * 3
        array3d=np.zeros(elemente)
        array3d=array3d.reshape(array2d.shape[0], array2d.shape[1], 3)
        print(array2d.shape)
        array1=np.where(array2d == 1, 1, array2d)
        array2=np.where(array2d == 2, 2, array2d)
        array3=np.where(array2d == 3, 3, array2d)
        # array3d=np.concatenate((array1, array2, array3))
        array3d[:,:,0] = array1
        array3d[:,:,1] = array2
        array3d[:,:,2] = array3
        return array3d

    def chains_to_contourcodes_2(self, chaincodes,\
                shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])):
        '''
        6.2 Converts a chain code to a contour code
         Input parameters:
             chaincodes - Chain code
             shift - Definition of the valid shift
         Return parameters:
             contourcodes - Contour code
        '''
        # shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])
        contourcodes = []
        for chain in chaincodes:
            n = len(chain)
            contour = np.zeros((n,2))
            contour[0] = [chain[0][0], chain[0][1]] # The starting point is adopted directly
            for idx in range(1,n):
                # Calculate coordinates
                contour[idx] = contour[idx - 1] + shift[chain[idx]]
            contourcodes.append(contour)
        return contourcodes         
