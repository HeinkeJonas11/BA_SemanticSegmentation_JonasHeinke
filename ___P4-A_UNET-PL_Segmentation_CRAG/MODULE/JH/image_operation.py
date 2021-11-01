# phyton3
# Letzte Änderung: 2021-02-12
# J.H

''' 
Inhalt
Klassen:
AddToImg - 1. Addiet diverse grafischen Elementen in ein Basisbild 
ImageInterface - 2. Zugriff auf Bilder und Bildbereiche
'''

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import __MODULE.JH.cam_objekte as camObj 
# import MODULE.JH.Utils as utils


class AddToImg():
    ''' KLASSE (https://www.hdm-stuttgart.de/~maucher/Python/html/Klassen.html)
    Einem Bild werden andere Bilder oder grafische Elemente überlagert.
    Das Ergebnis wird ausgegeben.
    Parameter:
        image   - Basisbild
        verbose - Steuerung der Ausgabe {True, False}'''
    # KONSTRUKTOR mit Attributen
    def __init__(self, image, verbose=False):
        super(AddToImg, self).__init__()
        self.image = image    # Basis- bzw. Ausgangsbild, Bild das ergänzt werden soll
        self.verbose=verbose  # um ggf. etwas auszugeben
    
    ### METHODEN ####
    def a_mask(self, mask):
        ''' Addiert zum Bild self.Images eine Maske
        Parameter:
            mask   - Maskembild zum Ausblenden von Objekten/Fahrzeugen
        Returm:
            image - Maskiertes Bild '''   
        mask_array = np.array(mask)
        masked = np.array(self.image)
        masked[mask_array > 0] = 255
        self.image=Image.fromarray(masked)
        self.image=Image.fromarray(masked)
        return self.image

    def attribut_boxes(self, proper):
        ''' Zeichnet für jedes Auto einen Rahmen/Box unter 
        Verwendung der Attribute/Eigenschaften in das Basisdaten
        Parameter:
            proper - Eigenschaften der Autos
                    Inhalt -> {imageID, typ,B_,C_yaw,A_,y_,z_,x_, i,
                    abstandToCam, winkelToCam, flaeche[i],
                    boxesArray[i][0],boxesArray[i][1],boxesArray[i][2],boxesArray[i][3]}
        Return:
            Image.fromarray(imgArray) - Bild mit Ergänzungen '''
        imgArray = np.array(self.image) # Image als np-array
        img_h,img_w,img_c = imgArray.shape 
        imageId,objIndex,x_, y_, z_, x1,y1,x2,y2=self.decompose_xyz(proper)
        # Beschriftung des Bildes mit ID-Nummer (links, oben)
        cv2.putText(imgArray, imageId, (0,60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5, color=(255,0,255), thickness = 3)
        # Projektion der 3D-Koordinaten in die 2D-Projektion
        xyz=[] # Neu ausrichten
        xyz.append((y_, z_, x_))
        x0,y0,_,_ = camObj.to_cam_xy(xyz)
        imgArray = cv2.circle(imgArray, (int(x0),int(y0)), 10, (0,255,255), -1) # Punkt entsprechend Basisdaten
        # Indizieren der Objekte im Bild
        cv2.putText(imgArray, str(objIndex), (int(x0),int(y0)), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,color = (0, 255, 255), thickness = 4 )
        point1 = (int(img_w * x1), int(img_h * y1))
        point2 = (int(img_w * x2), int(img_h * y2))
        w=point2[0] - point1[0]
        h=point2[1] - point1[1]
        imgArray=cv2.rectangle(imgArray, point1, point2, (255,0,0), 5)
        cv2.putText(imgArray, str(objIndex), point1, cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,color = (255, 0, 0), thickness = 4 )
        self.image=Image.fromarray(imgArray)
        return Image.fromarray(imgArray)    

    def center_points(self, proper):
        ''' Zeichnet für ein Auto einen Mittelpunkt auf Grundlage der Basisdaten
        Parameter:
            proper - Eigenschaften eines Autos
                    ZusammensInhalt -> {imageID, typ,B_,C_yaw,A_,y_,z_,x_, i,
                    abstandToCam, winkelToCam, flaeche[i],
                    boxesArray[i][0],boxesArray[i][1],boxesArray[i][2],boxesArray[i][3]}
        Return:
            self.image - Bild mit einer Ergänzung (Instanzvariable)'''
        _, _,x_, y_, z_,_,_,_,_=self.decompose_xyz(proper)
        imgArray = np.array(self.image) # Image als np-array
        img_h,img_w,img_c = imgArray.shape 
        _,_,x0,y0=camObj.p_xyz_to_cam_xy([[x_,y_,z_]])
        # Car-Koordinaten
        imgArray = cv2.circle(imgArray, (int(x0),int(y0)), 10, (255,0,255), -1)
        self.image=Image.fromarray(imgArray)
        return self.image

    def a_kss(self, D0=np.array([[-0.1, 0, 0]])):
        ''' Zeichnet ein Koordinatensystem
        Parameter:
            D0     - Verschiebung des 3D-Koordinatensystem zur verbesserten räumlichen Wahrnehmung in der 2D-Projektion
        Return:
            self.image - Bild mit dem Koordinatensystem '''
        imgArray = np.array(self.image) # Image als np-array
        img_h,img_w,img_c = imgArray.shape 
        # KKS x
        _,_,x0x,y0x=camObj.p_xyz_to_cam_xy(D0+ [[0,   0.4 ,  1]])
        _,_,x1x,y1x=camObj.p_xyz_to_cam_xy(D0+ [[ 0.1 ,  0.4,  1]])
        imgArray = cv2.arrowedLine(imgArray, (int(x0x),int(y0x)), (int(x1x),int(y1x)), (0,0,255), 12, tipLength=0.1)
        cv2.putText(imgArray, 'x', (x1x+5,y1x), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3,color = (0, 0, 255), thickness = 8 )
        # KKS A, Roll
        _,_,x0A,y0A=camObj.p_xyz_to_cam_xy(D0+ [[ 0.15, 0.4, 1]]) # Start 
        _,_,x1A,y1A=camObj.p_xyz_to_cam_xy(D0+ [[ 0.25, 0.4, 1]])   # End
        _,_,x2A,y2A=camObj.p_xyz_to_cam_xy(D0+ [[ 0.26, 0.4, 1]]) # wegen der Perspektive für den zweiten Pfeil ein Zusatzpunkt
        # zwei Pfeile für Rotation
        imgArray = cv2.arrowedLine(imgArray, (int(x0A),int(y0A)), (int(x1A),int(y1A)), (0,0,255), thickness=12, tipLength=0.15)
        imgArray = cv2.arrowedLine(imgArray, (int(x0A),int(y0A)), (int(x2A),int(y2A)), (0,0,255), thickness=12, tipLength=0.15)
        cv2.putText(imgArray, 'roll(A)', (x1A+55,y1A), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,color = (0, 0,255), thickness = 8 )
        # KKS y ------------------------------------------
        _,_,x0y,y0y=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4,   1]])
        _,_,x1y,y1y=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.5, 1]])
        imgArray = cv2.arrowedLine(imgArray, (int(x0y),int(y0y)), (int(x1y),int(y1y)), (0,255,0), 12, tipLength=0.1)
        cv2.putText(imgArray, 'y', (x1y+20,y1y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3,color = (0, 255,0), thickness = 8 )
        # KKS -B, Pitch
        _,_,x0B,y0B=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.35,   1]])
        _,_,x1B,y1B=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.27,   1]])
        _,_,x2B,y2B=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.26,   1]]) 
        imgArray = cv2.arrowedLine(imgArray, (int(x0B),int(y0B)), (int(x1B),int(y1B)), (0,255,0), thickness=12, tipLength=0.15)
        imgArray = cv2.arrowedLine(imgArray, (int(x0B),int(y0B)), (int(x2B),int(y2B)), (0,255,0), thickness=12, tipLength=0.15)
        cv2.putText(imgArray, 'pitch(-B)', (x1B-350,y1B), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,color = (0, 255,0), thickness = 8 )
        # KKS z ---------------------------------------------
        _,_,x0z,y0z=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4, 1  ]])
        _,_,x1z,y1z=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4, 1.1]])
        imgArray = cv2.arrowedLine(imgArray, (int(x0z),int(y0z)), (int(x1z),int(y1z)), (255,0,0), 12, tipLength=0.15)
        imgArray = cv2.circle(imgArray, (int(x0z),int(y0z)), 10, (255,0,0), -1) # Punkt für Ursprung
        cv2.putText(imgArray, 'z', (x1z+10,y1z), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3,color = (255, 0,0), thickness = 8 )
        # KKS C, Yaw
        _,_,x0C,y0C=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4,   1.2]])
        _,_,x1C,y1C=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4,   1.5]])
        _,_,x2C,y2C=camObj.p_xyz_to_cam_xy(D0+ [[0, 0.4,   1.55]]) # wegen der Perspektive für den zweiten Pfeil ein Zusatzpunkt
        imgArray = cv2.arrowedLine(imgArray, (int(x0C),int(y0C)), (int(x1C),int(y1C)), (255,0,0), thickness=12, tipLength=0.15)
        imgArray = cv2.arrowedLine(imgArray, (int(x0C),int(y0C)), (int(x2C),int(y2C)), (255,0,0), thickness=12, tipLength=0.15)
        cv2.putText(imgArray, 'yaw(C)', (x2C+15,y2C), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2,color = (255,0,0), thickness = 8 )
        self.image=Image.fromarray(imgArray)
        return self.image

    ### METHODEN ZUR LOKALEN VERWENDUNG ###
    ### Methoden zum Umwandeln der Eigenschaftsliste in namensspezifische Variablen.
    ### (Damit soll eine verbesserte Lesbarkeit der Methoden/Programme erreicht werden.

    def decompose_xyz(self, proper):
        ''' Listenelemente benennen
        Parameter:
            prop - Eigenschaften/Attribute der Autos eines Bildes als Liste
        Return: - Eigenschaften mit spezifischen Namen
            imageId - ID des Bildes
            objIndex - Index eines Fahrzeuges eines Bildes
            x_, y_, z_  - 3D-Koordinaten des Fahrzeuges
            x1,y1,x2.y2 - Diagonalkoordinaten des Rechtecks auf dem Bild'''
        imageId=proper[0]
        objIndex=int(proper[1])
        x_=proper[6]    #y_=proper[6]
        y_=proper[7]    #z_=proper[7]
        z_=proper[8]    #x_=proper[8]
        x1=proper[13]   # Box-Koordinaten
        y1=proper[14]
        x2=proper[15]
        y2=proper[16]
        return imageId,objIndex,x_, y_, z_, x1,y1,x2,y2
   
    def decompose(self, proper):
        ''' Alle Listenelemente benennen
        Parameter:
            proper - Eigenschaften/Attribute der Autos eines Bildes 
       Return: - Eigenschaften mit spezifischen Namen
            imageId - ID des Bildes
            objIndex - Index eines Fahrzeuges des Bildes
            carTyp   - Fahrzeugtyp codiert (0 - 79)
            c_, b_, a_ - yaw, pitch, roll des Fahrzeuges im 3D-System
            x_, y_, z_  - 3D-Koordinaten des Fahrzeuges
            x1,y1,x2.y2 - Diagonalkoordinaten des Rechtecks auf dem Bild
            boxindex = objIndex - Index eines Fahrzeuges des Bildes
            abstandToCam - Abstand des Fahrzeuges zur Kamer im 2D-Bild (0 -> 1)
            winkelToCam  - Winkel zum Fahrzeu als Winkelfunktioswer (0 -> 1)
            flaeche      - Rechteckfläche
            x1,y1,x2.y2 - Diagonalkoordinaten des Rechtecks auf dem Bild'''
        imageId=proper[0]
        objIndex=int(proper[1])
        cartype=proper[2] 
        c_=proper[3]    #B_=proper[3]
        b_=proper[4]    #C_yaw=proper[4]
        a_=proper[5]    #A_=proper[5]
        x_=proper[6]    #y_=proper[6]
        y_=proper[7]    #z_=proper[7]
        z_=proper[8]    #x_=proper[8]
        boxindex=int(proper[9]) # wie objIndex
        abstandToCam=proper[10]
        winkelToCam=proper[11]
        flaeche=proper[12]
        x1=proper[13]
        y1=proper[14]
        x2=proper[15]
        y2=proper[16]
        return imageId, objIndex, cartype, c_, b_, a_, x_, y_, z_, boxindex, abstandToCam, winkelToCam, flaeche, x1,y1,x2,y2

########################################################################################################

class ImageInterface():
    ''' (https://www.hdm-stuttgart.de/~maucher/Python/html/Klassen.html)
    2. Einem Bild werden andere Bilder oder grafische Elemente überlagert.
        Das Ergebnis wird ausgegeben.
    Parameter:
        imgWeite - Weite des Basisbildes in Pixeln
        imgHoehe - Höhe des Basisbildes in Pixeln
        verbose  - Steuerung der Ausgabe zur Kontroll '''
    def __init__(self, image, verbose=False):
        '''Konstruktor '''
        super(ImageInterface, self).__init__()
        self.image=image            # Bild
        self.imgHoehe, self.imgWeite =image.size      # Höhe der Basisbilder in Pixel
        self.verbose=verbose        # um ggf. etwas auszugeben

    ### region: METHODEN ####
    def cut_sub_image(self, dirImg, imageId, x1,y1,x2,y2):
        ''' Bildbereich ausschneiden
        Parameter:
            imageId - Id der Straßenszene
            x1,y1,x2,y2 - Diagonal-Koordinaten des auszuschneidenden Bereichs (Box, Subbild)
        Return:
            subImage - Bildausschnitt
            boxInPixel  (x1,y1,x2,y2) - Diagonalkoordinaten in Pixel'''
        image= Image.open(dirImg + imageId+'.jpg')
        hoehe, weite =image.size  
        weite, hoehe=image.size   ##### Basisbildgröße neu ermitteln #####
        boxInPixel=(int(weite * x1),int(hoehe * y1),    
                    int(weite * x2),int(hoehe * y2))
        subImage=image.crop(boxInPixel)
        return subImage, boxInPixel

    
    def cut_sub_image_u1(self, image, x1,y1,x2,y2):
        ''' Bildbereich ausschneiden (Überladung)
        Parameter:
            image - Straßenszene als Bild
            x1,y1,x2,y2 - Diagonal-Koordinaten des auszuschneidenden Bereichs (Box, Subbild)
        Return:
            subImage - Bildausschnitt
            boxInPixel  (x1,y1,x2,y2) - Diagonalkoordinaten in Pixel'''
        hoehe, weite =image.size  
        weite, hoehe=image.size   ##### Basisbildgröße neu ermitteln #####
        boxInPixel=(int(weite * x1),int(hoehe * y1),    
                    int(weite * x2),int(hoehe * y2))
        subImage=image.crop(boxInPixel)
        return subImage, boxInPixel

    
    def cut_sub_image_u2(self, x1,y1,x2,y2):
        ''' Bildbereich ausschneiden (Überladung)
        Parameter:
            self.image  - Instanzvariable der Straßenszene als Bild
            x1,y1,x2,y2 - Diagonal-Koordinaten des auszuschneidenden Bereichs (Box, Subbild)
        Return:
            subImage - Bildausschnitt
            boxInPixel  (x1,y1,x2,y2) - Diagonalkoordinaten in Pixel'''
        self.imgWeite, self.imgHoehe=self.image.size   ##### Basisbildgröße neu ermitteln #####
        boxInPixel=(int(self.imgWeite * x1),int(self.imgHoehe * y1),    
                    int(self.imgWeite * x2),int(self.imgHoehe * y2))
        subImage=self.image.crop(boxInPixel)
        return subImage, boxInPixel

    ''' ++++++++++++++++++++++++++++++ '''
    
    def show_np_image(self, npImage, bezeichnung='Bild'):
        ''' Zeige ein numphy-Image (not in use)
        Parameter: 
            npImage     - np-Array als Bild
            bezeichnung - String zur Bezeichnung des Bildes '''         
        fig, ax = plt.subplots(figsize=(18,26))
        ax.imshow(npImage) 
        ax.set_title(bezeichnung)
        ax.axis('off')
    ''' +++++++++++++++++++++++++++ '''
    
    def get_abstand_winkel(self, boxes):
        ''' Ermittelt Eigenschaften aller Pixel eines Subbbildes in Relation zur Camera.
        Speichert die Werte in einem Array, die als zusätzliche Kanäle des Bildes dienen
        Parameter:
            boxes   - Koordinaten der Umrandungen aller gefundenen Objekte
        Return (Datenarrays):
            abstandFromCam <np.array> - Abstand aller Bildpunkte eines Ausschnitts zur Kamera in 0 ..1
            richtungFromCam <np.array>- Richtung aller Bildpunktes eines Ausschnitts von der Kamera ausgesehen'''
        x0=int(0.5 * self.imgWeite) # Kammera-Bezugspunkt (mitte, unten)
        y0=int(1   * self.imgHoehe) 
        # Koordinaten der Box-Diagonalpunkte
        x1= boxes[0]
        y1= boxes[1]
        x2= boxes[2]
        y2= boxes[3]
        # print (x1,y1,x2,y2)
        W= x2-x1
        H=y2-y1 
        # Maximale Entfernung eines Bildpunktes zur Cam
        abstandMax=np.sqrt(self.imgWeite*self.imgWeite/4+self.imgHoehe*self.imgHoehe)
        #print(W,H)
        abstandFromCam=np.zeros((H,W,1))
        richtungFromCam=np.zeros((H,W,1))
        for y in range(0,H,1):
            for x in range(0,W,1):
                # normierter Abstand 0..1
                abstand= np.sqrt(((x1+x)-x0)*((x1+x)-x0)+((y1+y)-y0)*((y1+y)-y0)) # in Pixel
                abstandFromCam[y][x][0]= abstand/abstandMax # in 0 .. 1
                # Richtung in rad
                ##richtungRad=np.arcsin(((x1+x)-x0)/abstand) # in rad
                ##richtungRadFromCam[y][x][0]= richtungRad # in rad #radToGrd(richtung)
                # RichtungsSinus normiert 0..1
                richtungSin= ((x1+x)-x0)/abstand # Werte zwischen -1 ... +1
                richtungNorm= (richtungSin+1)/2
                richtungFromCam[y][x][0]= richtungNorm # normalisiert 0..1
        return abstandFromCam, richtungFromCam
    
