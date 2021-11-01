# phyton3
# Letzte Änderung: 2021
# Jonas Heinke

''' Inhalt

Path   - 3. Pfadangaben
'''
import pathlib
import os
import glob


class Path():
    ''' 3. Pfadangaben '''
    # Liefert das Projektverzeichnis
    project= pathlib.Path.cwd()
    # Basispfad des Datensets
    # dataset = pathlib.Path.cwd() / 'CRAG_v2/CRAG'
    dataset = project / 'CRAG_v2/CRAG/'
    # Pfad für trainierte Modelle
    model= project / 'models/'
    # ----------------------------
   

    def get_filenames(self, path: pathlib.Path, dateifilter: str = '*.png'):
        ''' HILSMETHODE [in Anlehnung an https://johschmidt42.medium.com/]: 
        Liefert eine Liste der Dateien eines Verzeichnisses systemunabhängig.
        Eingang:
            path     - Pfad in dem die Dateien gesucht werden.
            dateifilter - Filter zur Auswahl der Dateien, Standard: *.png
        Rückgabe:
            list_of_filenames - Liste der Dateinamen
        '''
        list_of_filenames = [file for file in path.glob(dateifilter) if file.is_file()]
        return list_of_filenames








        
	


