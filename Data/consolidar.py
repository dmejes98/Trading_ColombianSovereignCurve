# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:40:16 2023

@author: David Mejia
"""

import pandas as pd
import numpy as np
import time
import os

inicio = time.time()

# En esta carpeta se encuentran los libros de excel mensuales de todas las operaciones del SEN.
nombre_carpeta = "C:/Users/David Mejia/Documents/EAFIT/Tesis/Trading the Curve/Data/SEN"

contenido = os.listdir(nombre_carpeta)

operaciones = pd.DataFrame(columns=[])
for elemento in contenido:

    libro = pd.read_excel(rf'C:\Users\David Mejia\Documents\EAFIT\Tesis\Trading the Curve\Data\SEN\{elemento}', header=[9])
    
    # Retiramos TUVT
    aux = []
    for i in libro.index:
        
        aux.append(libro.iloc[i,4][0:4])
    
    del i
    
    libro["aux"] = aux
    
    libro = libro[libro["aux"] != ("TUVT" and "TFVT")]
    
    #Retiramos Operaciones de Liquidez
    libro = libro[libro["* VR. NOMINAL COLATERAL"] == 0]
    libro.columns
    libro.drop(["aux", "* VR. NOMINAL COLATERAL", "PLAZO (De regreso para SIML, Repos e INTB)", "* TASA/ PRECIO\nCOLATERAL", "* TASA/ PRECIO\nEQUIV.\nCOLATERAL"], axis = 1, inplace = True)
    
    operaciones = pd.concat([operaciones, libro])

operaciones.reset_index(inplace = True)
operaciones.drop(["index"], axis = 1, inplace = True)

final = time.time()

ejec_time = (final - inicio)/60
print("Tiempo de ejecución en minutos: ", ejec_time)

operaciones.to_excel('SEN.xlsx')
