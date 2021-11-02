# **Projekte zur semantischen Segmentation von Darmdrüsen in der Histopathologie**

Die Projekte befinden sich auf Google-Drive und auf GitHub.

Google Drive: https://drive.google.com/drive/folders/1OmN6ZcC_s2Uq1TyOrsZ1PBm9WkFkTDyv?usp=sharing

GitHub .... : https://github.com/HeinkeJonas11/BA_SemanticSegmentation_JonasHeinke

Mit Google Colab können die Projekte cloudbasiert von Google-Drive aus gestartet und abgearbeitet werden. Die kostenfreie Version von Google Colab ist allerdings limitiert. Benötigt wird für anspruchsvolle Trainnings die kostenpflichtige Version „Google+“ für zusätzliche Ressourcen. Das betrifft eine leistungsstarke GPU.

Das Datenset CRAG_v2 muss separat heruntergeladen und in das zugehörige Verzeichnis "../___Datasets/" kopiert werden:
https://drive.google.com/file/d/1p3dZXpgeA1IcGO6vXhStbVLMku-fZTmQ/view

Der Download des Datensets WARWICK-QU erfolgt automatisch.
#
### Die Projekte verwenden Programmteile aus:

[1] H. Pinckaers, „Software: Neural Ordinary Differential Equations for Semantic Segmentation of Individual Colon Glands,“ GitHub, 09. 04. 2019. [Online]. Available: https://github.com/DIAGNijmegen/neural-odes-segmentation.

[2] J. Schmidt, „Creating and training a U-Net model with PyTorch for 2D & 3D semantic,“ Towards Data Science, 2020. [Online]. Available: https://johschmidt42.medium.com/.

[3] ELEKTRONN-Team und J. Kornfeld (Berater), „Software: elektronn3,“ GitHub, 2017 (2020). [Online]. Available: https://github.com/ELEKTRONN/elektronn3. 
#
### **Kurzzeichen, Verzeichnisname und Beschreibung**
#
### P3-A ....................... ___P3-A_UNET_RESNET_NODE-PL-Segmenation_QU
Originalprojekt von Pinkers mit geringfügigen, eigenen Erweiterungen zur Ergebnisprotokollierung, zur klassenbezogenen Bewertung und zur Visualisierung der Bilder, Masken und Prognosen

### P4 .........................___P4_neural-odes-segmentation-master_CRAG
Projekt von Pinckaers zur Segmentation von Drüsen des Datensets CRAG_v2 mit notwendigen Anpassungen zum Einlesen der Daten.

### P4-A ..................... ___P4-A_UNET-PL-Segmentation_CRAG
Projekt von Pinckaers zur Segmentation von Drüsen. Eine Anpassung dient zum Einlesen des Datensets CRAG_v2. Erweiterungen sind die Berechnung von klassenbezogenen Kennzahlen, die Visualisierung von Bildern und die Protokollierung von Segmentationsergebnissen.

### P5 .......................... ___P5_Eigenes_Projekt_Segmentation_CRAG
Eigenes Projekt zur Segmentation des Datensets CRAG_v2, bestehend aus mehreren Teilprojekten:

P5-01 – Visualisierung des Datensets

P5-02 – Zerschneiden großer Bilder (optional, siehe P5-A)

P5-03 – Training mit Validierung

P5-04 – Prognose und Nachbearbeitung

P5-05 – Bewertung und Visualisierung

Das U-Net basiert auf Veröffentlichungen von Schmidt und ELEKTRONN-Team. Zur Nachbearbeitung werden morphologische Operation, der Kettencode oder Module von Python „skimage“ zur Indizierung der Drüsen verwendet. Die Parametrierung erfolgt mithilfe einer Konfigurationsdatei. Zur Bewertung der Prognoseabweichung dienen die Kennzahlen: Der Dice-Index, der F1-Score und der Weighted-Shape.

### P7 ........................ ___P7_Eigenes_Projekt_Segmentation_QU
Das Projekt ist identisch dem Projekt P5, bis auf das Einlesen der Bild-Masken-Paare des Datensets „WARWICK-QU”.
#


