# image processing
# phyton3
# Letzte Änderung: 2021-06-08
# J.H

'''
Inhalt
Klassen:
IdentifyObject - 1. Bild-Array-Operationen

    Methoden:
    chaincode      - 1.1 Liefert den Kettencode aller Objekte eines Arrays. Jedes Objekt erhält eine eindeutige
                    Identifikationsnummer. Zusätzlich wird eine Array mit den identifizierten Objekten erstellt.
    padding_array - 1.2 Fügt einem Array einen Rand hinzu.
                    Wird benötigt, um Elemente, die den rand tangieren eindeutig zu bestimmen.
    unpadding_array - 1.3. Entfernt Randzeilen und Randspalten von einem Array
    unpadding_chaincode - 1.4 Verschiebt den Startpunkt des Kettecods um eine Zeile nach oben und eine Spalte
                         nach links. Das ist notwendig, wenn der Kettencode für Objekte des Arrays mit Rand
                         ermittelt wurde.
    fill_object  - 1.5 Füllt umrandetes Objekt eines Arrays mit der anggebenen Id
# ------------------
DrawInArray() -  2. Zeichnet in ein Array, zum Beispiel eine Kontur

    Methoden:
    a_chaincode - 2.1 Verwendet den Kettencode um eine Kontur in eine Array zu übertragen.
# ------------------------------------    
Convert() -  3. Konvertiert Arrays und Images
             IN WORK UND NICHT IN NUTZUNG
# -----------------------------------------------                             
'''

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


''' Auch in dem Modul image_prozessing'''
class IdentifyObject():
    ''' 
    1. Klasse Bild-Array-Operationen
    '''
    #Konstruktor
    def __init__(self, inputarray, verbose=False):
        super(IdentifyObject, self).__init__()
        # Eingangsbild als zweidimensionales Nupy-Array
        self.array = inputarray    
        self.array = self.padding_array(self.array)
        # Zur Ausgabe zwecks Dokumentation und zum Testen
        self.verbose=verbose  
        # Enthält alle bereits gefundenen Objekte der jeweiligen Objektid.
        #  Anfangswerte sind 0, wie Hintergrund.
        self.chain_array= np.zeros(self.array.shape).astype(np.int16)

    def chaincode(self, threshold, idx=0):
        '''
        1.1 Liefert den Kettencode aller Objekte eines Arrays. Jedes Objekt erhält eine eindeutige
        Identifikationsnummer. Zusätzlich wird eine Array mit den identifizierten Objekten erstellt.
        Für diese Objekte werden ein Start- beziehungsweise ein Anfangspunkt gesucht.
        Von dort aus beginnt und endet die Umrandung des Objektes. Die Umrandeten Flächen
        werden zusätzlich ausgefüllt (siehe Rückgabeparameter).
        Hinweis zu den Indizes der Arrays: array[i, j] i - Zeilen, j - Spalten 
        Eingang Klassenvarible:
            self.array - Array mit Objekten
        Eingangsparameter:
            idx       - Array-Index, dient nur im Falle eines Fehls zur Kennung
            threshold - Elemente mit dem Intensitätswert werden als Objekt betrachttet.
                        (Es ist ein Wert != 0, meist == 1)
        Rückgabeparameter:
            chaincodes - Liste der Kettencods 
                Beispiel für einen Listeneintrag: 
                chaincodes[0]=[[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
            self.chain_array - Array gleichartiger Objekten. Jedes Objekt besitzt eine
                eindeutige Id. Alle Elemente eines Objektes besitzen die gleiche ID.
        '''
        start_chain=np.array([0,0])
        neighbor=np.array([0,0])
        point_of_chain=np.array([0,0])
        id=1
        chaincodes=[]
        #chain=[] # für alle Fälle hier schon definiert
        gefunden=False

        while True:
            print('Id', id, end=', ')
            for i in range(start_chain[0], self.array.shape[0]):
                if gefunden: # Damit beide Schleifen verlassen werden können
                    break
                for j in range(0, self.array.shape[1]):
                    # SUCHE DEN STARTPUNKT EINES OBJEKTES
                    # Suche ein Objekt mit der Intensität threshold,
                    # dass noch nicht gefunden wurde.
                    if self.array[i,j] == threshold and self.chain_array[i, j]==0:
                        # Startpunkt gefunden
                        chain=[]
                        start_chain=[i, j]
                        chain.append(start_chain)
                        self.chain_array[i,j]=id
                        gefunden=True
                        break
                        # --- Startpunkt eines neuen Objektes gefunden ---

            # if id==13: break ### für Test eine Unterbrechung

            # VERFOLGE JETZT DIE KONTUR
            # Nachbarindex und Verschiebung in i,j
            shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])
            # Bei diesem Nachbarn 4 beginnt die Verfolgungssuche. Es ist kein
            # Objektpunkt, da links neben der gefundenen Kontur.
            point_of_chain=start_chain
            s=0
            s_check=s
            zyklus=0
            while True:
                
                # Gesucht wird der Intensitätsübergang von Threshold (schwarz) 
                # nach nicht Threshold (weiß)
                # neighbor= point_of_chain + size[i % 8]
                while True:
                    s +=1
                    zyklus +=1
                    # i wächst stätig um 1, muss aber in den intervall [0,8] zuückgeführt werden
                    s_check = s % 8
                    neighbor= point_of_chain + shift[s_check]
                    if self.verbose: #----------------------------------------------------------
                        print('vor : ',s ,s_check, neighbor, point_of_chain, start_chain,'neighborwert= ',self.array[neighbor[0], neighbor[1]])
                    if self.array[neighbor[0], neighbor[1]] == threshold or zyklus >8:
                        zyklus=0
                        break
                ''' while_end zum Finden eines Randpunktes'''        
                # Rand-, Konturpunkt gefunden #***********************************************************
                try:
                    chain.append(s_check)
                except:
                    print(f'Kein Startpunkt? Kein Objekt in Array[{idx}] gefunden!')
                    
                self.chain_array[neighbor[0], neighbor[1]]=np.uint8(id)
                # gehe je nach gefundenen Nachbarn um ein oder zwei Schritte zurück
                s -=3
                if neighbor[0] !=  point_of_chain[0] and neighbor[1] != point_of_chain[1]:
                    s -=1
                    if self.verbose:
                        print('diagonaler Anschluss)')
                else:
                    if self.verbose:
                        print('gerader Anschluss!')                    
                point_of_chain=neighbor
                if point_of_chain[0] == start_chain[0] and point_of_chain[1] == start_chain[1]:
                    break
                ''' while_end zum Umranden einer Kontur mit der id '''
            # Ausfüllen der Objektfläche
            self.chain_array=self.fill_object(np.uint8(id), self.chain_array )
            # Übernehme den Kettencode in die Liste
            try:
                chaincodes.append(chain)
            except:
                print('Kein Kettencode zum anhängen!')
            # Suche für nächste Kontur vorbereiten
            id +=1
            s= s % 8 # Gefahr vermeiden
            gefunden=False
            # if start_chain[1]==self.array.shape[1]-1 and start_chain[0]==self.array.shape[0]-1:
            if j==self.array.shape[1]-1 and i==self.array.shape[0]-1:
                 break
            ''' while_end zum Auffinden aller Objekte eines Bildes für einen vorgegeben Intensitätswert '''    
        self.chain_array= self.unpadding_array(self.chain_array)
        chaincodes=self.unpadding_chaincode(chaincodes)
        return chaincodes, self.chain_array

    def padding_array(self, array):
        '''
        1.2 Fügt einem Array einen Rand hinzu.
        Eingang:
            array - zweidimensionales Array
        Rückgabe:
            array_padding - Array mit zusätzlichem rand
        '''
        array_padding=np.zeros((array.shape[0]+2, array.shape[1]+2))
        array_padding[1:-1,1:-1]=array[:,:]
        return array_padding

    def unpadding_array(self, array_padding):
        '''
        1.3 Entfernt Randzeilen und Randspalten von einem Array
        Eingang:
            array_padding - Zweidimensionales Array, dessen Rand entfernt werden soll
        Rückgabe:
            array - Array jetzt ohne Rand
        '''
        array=np.zeros((array_padding.shape[0]-2, array_padding.shape[1]-2))
        # print(self.array.shape, array.shape)
        array[:,:] = array_padding[1:-1,1:-1]
        return array

    def unpadding_chaincode(self, chaincodes):
        '''
        1.4 Verschiebt den Startpunkt des Kettecods um eine Zeile nach oben und eine Spalte nach links.
        Das ist notwendig, da der kettenkode für Objekte des Array mit Rand ermittelt wurde.
        Eingang:
            chaincodes - Liste der Kettencode
            Beispiel für einen Listeneintrag: [[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
        Rückgabe:
            chaincodes - Liste der korrigierten Kettencode
        '''
        for id in range(len(chaincodes)):
           
                chaincodes[id][0][0] -= 1
                chaincodes[id][0][1] -= 1
                  
        return chaincodes

    def fill_object(self, id, chain_array):
        '''
        1.5 Füllt umrandete Objekte des self.chain_array mit der anggeben id
            Eingang:
                id - Identifikationsnummer des Objektes, Objekt ist bereits umrandet.
                     Mit dieser Ud wird die Objektfläche gefüllt.
                chain_array - Array mit n Objekten. Jedes Objekt besitzt eine eigenen Id,
                              dessen Rand markiert ist. 
            Rückgabe:
                chain_array - Array mit n Objekten. Die innere Fläche ist mit der 
                              zugehörigen Id gefüllt.
        '''
        for i in range(chain_array.shape[0]):
            for j in range(chain_array.shape[1]):
                # ist links, rechts, oben und unten eine Grenze des Objektes?
                up   = len(np.where(chain_array[0: i, j] == id)[-1])
                down = len(np.where(chain_array[i: , j] == id)[-1])
                left = len(np.where(chain_array[i,  0:j] == id)[-1])
                right= len(np.where(chain_array[i, j: ] == id)[-1] )
                # print('len ....', up, down, left, right)
                if up>0 and down>0 and left>0 and right>0:
                    chain_array[i,j]= id
        return chain_array

#class ImageArrayOperation(): EHEMALS
class IdentifyObject_2():
    ''' 
    1. Klasse Bild-Array-Operationen
    '''
    #Konstruktor
    def __init__(self, inputarray, verbose=False):
        super(IdentifyObject, self).__init__()
        # Eingangsbild als zweidimensionales Nupy-Array
        self.array = inputarray    
        self.array = self.padding_array(self.array)
        # Zur Ausgabe zwecks Dokumentation und zum Testen
        self.verbose=verbose  
        # Enthält alle bereits gefundenen Objekte der jeweiligen Objektid.
        #  Anfangswerte sind 0, wie Hintergrund.
        self.chain_array= np.zeros(self.array.shape).astype(np.int16)

    def chaincode(self, threshold):
        '''
        1.1 Liefert den Kettencode aller Objekte eines Arrays. Jedes Objekt erhält eine eindeutige
        Identifikationsnummer. Zusätzlich wird eine Array mit den identifizierten Objekten erstellt.
        Für diese Objekte werden ein Start- beziehungsweise ein Anfangspunkt gesucht.
        Von dort aus beginnt und endet die Umrandung des Objektes. Die Umrandeten Flächen
        werden zusätzlich ausgefüllt (siehe Rückgabeparameter).
        Hinweis zu den Indizes der Arrays: array[i, j] i - Zeilen, j - Spalten 
        Eingang Klassenvarible:
            self.array - Array mit Objekten
        Eingangsparameter:
            threshold - Elemente mit dem Intensitätswert werden als Objekt betrachttet.
                        (Es ist ein Wert != 0, meist == 1)
        Rückgabeparameter:
            chaincodes - Liste der Kettencods 
                Beispiel für einen Listeneintrag: 
                chaincodes[0]=[[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
            self.chain_array - Array gleichartiger Objekten. Jedes Objekt besitzt eine
                eindeutige Id. Alle Elemente eines Objektes besitzen die gleiche ID.
        '''
        start_chain=np.array([0,0])
        neighbor=np.array([0,0])
        point_of_chain=np.array([0,0])
        id=1
        chaincodes=[]
        chain=[] # für alle Fälle hier schon definiert
        gefunden=False

        while True:
            print('Id', id, end=', ')
            for i in range(start_chain[0], self.array.shape[0]):
                if gefunden: # Damit beide Schleifen verlassen werden können
                    break
                for j in range(0, self.array.shape[1]):
                    # SUCHE DEN STARTPUNKT EINES OBJEKTES
                    # Suche ein Objekt mit der Intensität threshold,
                    # dass noch nicht gefunden wurde.
                    if self.array[i,j] == threshold and self.chain_array[i, j]==0:
                        # Startpunkt gefunden
                        chain=[]
                        start_chain=[i, j]
                        chain.append(start_chain)
                        self.chain_array[i,j]=id
                        gefunden=True
                        break
                        # --- Startpunkt eines neuen Objektes gefunden ---

            # if id==13: break ### für Test eine Unterbrechung

            # VERFOLGE JETZT DIE KONTUR
            # Nachbarindex und Verschiebung in i,j
            shift=np.array([[-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,+1], [0,+1], [-1,+1]])
            # Bei diesem Nachbarn 4 beginnt die Verfolgungssuche. Es ist kein
            # Objektpunkt, da links neben der gefundenen Kontur.
            point_of_chain=start_chain
            s=0
            s_check = s
            zyklus=0
            while True:
                
                # Gesucht wird der Intensitätsübergang von Threshold (schwarz) 
                # nach nicht Threshold (weiß)
                # neighbor= point_of_chain + size[i % 8]
                while True:
                    s +=1
                    zyklus +=1
                    # i wächst stätig um 1, muss aber in den intervall [0,8] zuückgeführt werden
                    s_check = s % 8
                    neighbor= point_of_chain + shift[s_check]
                    if self.verbose: #----------------------------------------------------------
                        print('vor : ',s ,s_check, neighbor, point_of_chain, start_chain,'neighborwert= ',self.array[neighbor[0], neighbor[1]])
                    if self.array[neighbor[0], neighbor[1]] == threshold or zyklus >8:
                        zyklus=0
                        break
                ''' while_end zum Finden eines Randpunktes'''        
                # Rand-, Konturpunkt gefunden         
                try:
                    chain.append(s_check)
                except:
                    print(f'Kein Startpunkt? Kein Objekt im Image[{idx}] gefunden!')
                self.chain_array[neighbor[0], neighbor[1]]=np.uint8(id)
                # gehe je nach gefundenen Nachbarn um ein oder zwei Schritte zurück
                s -=3
                if neighbor[0] !=  point_of_chain[0] and neighbor[1] != point_of_chain[1]:
                    s -=1
                    if self.verbose:
                        print('diagonaler Anschluss)')
                else:
                    if self.verbose:
                        print('gerader Anschluss!')                    
                point_of_chain=neighbor
                if point_of_chain[0] == start_chain[0] and point_of_chain[1] == start_chain[1]:
                    break
                ''' while_end zum Umranden einer Kontur mit der id '''
            # Ausfüllen der Objektfläche
            self.chain_array=self.fill_object(np.uint8(id), self.chain_array )
            # Übernehme den Kettencode in die Liste
            try:
                chaincodes.append(chain)
            except:
                print('Kein Kettencode zum anhängen!')
            # Suche für nächste Kontur vorbereiten
            id +=1
            s= s % 8 # Gefahr vermeiden
            gefunden=False
            # if start_chain[1]==self.array.shape[1]-1 and start_chain[0]==self.array.shape[0]-1:
            if j==self.array.shape[1]-1 and i==self.array.shape[0]-1:
                 break
            ''' while_end zum Auffinden aller Objekte eines Bildes für einen vorgegeben Intensitätswert '''    
        self.chain_array= self.unpadding_array(self.chain_array)
        chaincodes=self.unpadding_chaincode(chaincodes)
        return chaincodes, self.chain_array

    def padding_array(self, array):
        '''
        1.2 Fügt einem Array einen Rand hinzu.
        Eingang:
            array - zweidimensionales Array
        Rückgabe:
            array_padding - Array mit zusätzlichem rand
        '''
        array_padding=np.zeros((array.shape[0]+2, array.shape[1]+2))
        array_padding[1:-1,1:-1]=array[:,:]
        return array_padding

    def unpadding_array(self, array_padding):
        '''
        1.3 Entfernt Randzeilen und Randspalten von einem Array
        Eingang:
            array_padding - Zweidimensionales Array, dessen Rand entfernt werden soll
        Rückgabe:
            array - Array jetzt ohne Rand
        '''
        array=np.zeros((array_padding.shape[0]-2, array_padding.shape[1]-2))
        # print(self.array.shape, array.shape)
        array[:,:] = array_padding[1:-1,1:-1]
        return array

    def unpadding_chaincode(self, chaincodes):
        '''
        1.4 Verschiebt den Startpunkt des Kettecods um eine Zeile nach oben und eine Spalte nach links.
        Das ist notwendig, da der kettenkode für Objekte des Array mit Rand ermittelt wurde.
        Eingang:
            chaincodes - Liste der Kettencode
            Beispiel für einen Listeneintrag: [[0, 6], 4, 4, 6, 5, 6, 6, 7, 6, 6, 0, 0, 2, 2, 2, 3, 1, 2, 2]
        Rückgabe:
            chaincodes - Liste der korrigierten Kettencode
        '''
        for id in range(len(chaincodes)):
            chaincodes[id][0][0] -= 1
            chaincodes[id][0][1] -= 1
        return chaincodes

    def fill_object(self, id, chain_array):
        '''
        1.5 Füllt umrandete Objekte des self.chain_array mit der anggeben id
            Eingang:
                id - Identifikationsnummer des Objektes, Objekt ist bereits umrandet.
                     Mit dieser Ud wird die Objektfläche gefüllt.
                chain_array - Array mit n Objekten. Jedes Objekt besitzt eine eigenen Id,
                              dessen Rand markiert ist. 
            Rückgabe:
                chain_array - Array mit n Objekten. Die innere Fläche ist mit der 
                              zugehörigen Id gefüllt.
        '''
        for i in range(chain_array.shape[0]):
            for j in range(chain_array.shape[1]):
                # ist links, rechts, oben und unten eine Grenze des Objektes?
                up   = len(np.where(chain_array[0: i, j] == id)[-1])
                down = len(np.where(chain_array[i: , j] == id)[-1])
                left = len(np.where(chain_array[i,  0:j] == id)[-1])
                right= len(np.where(chain_array[i, j: ] == id)[-1] )
                # print('len ....', up, down, left, right)
                if up>0 and down>0 and left>0 and right>0:
                    chain_array[i,j]= id
        return chain_array

#-------------------------------------------------------------------------

class DrawInArray():
    ''' 2. Zeichnet in ein Array, hier eine Kontur
    HINWEIS: Schnellere Berechnung wenn Klasse im Notebook'''
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
        Eingang Klassenvariablen:
            self.array - Array, dass verändert wird
        Eingang Methode:
            chaincode - Kettencode entsprechend der Definition in self.shift
            element_value - Wert, den die Konturpixel erhalten
        Rückgabe:
            self.array - Array mit Kontur entsprechend des Kettencods
        '''
        pi=int(chaincode[0][0])
        pj=int(chaincode[0][1])
        self.array[pi, pj]=element_value # Startpunktvalue
        if self.verbose:
            print(f'\na_chaincode -> Startpunkt: {pi},{pj}')
        for cod in range(1, len(chaincode)):
            i=int(self.shift[chaincode[cod]][0])
            j=int(self.shift[chaincode[cod]][1])
            pi+=i
            pj+=j
            self.array[pi, pj]=element_value #element_value
            if self.verbose:
                print(f'P: {pi}, {pj}',end=','  )
        return self.array 
    
    def get_array(self):
        return self.array


# ----------------------------------------------------------------------
class Convert():
    '''  
       3. Konvertiert Arrays und Images
            IN WORK
    '''
        #Konstruktor
    def __init__(self):
        super(Convert, self).__init__()

    def to_3d_array(self, array2d, dtype=int):
        # # IN WORK
        elemente=array2d.shape[0]*array2d.shape[1]*3
        array3d=np.zeros(elemente)
        array3d=array3d.reshape(array2d.shape[0], array2d.shape[1], 3)
        print(array2d.shape)
        array1=np.where(array2d == 1, 1, array2d)
        array2=np.where(array2d == 2, 2, array2d)
        array3=np.where(array2d == 3, 3, array2d)
        # array3d=np.concatenate((array1, array2, array3))
        array3d[:,:,0]=array1
        array3d[:,:,1]=array2
        array3d[:,:,2]=array3
        return array3d


    
        
          




    





                




  





            

