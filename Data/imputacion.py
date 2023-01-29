# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:51:00 2023

@author: David Mejia
"""

import pandas as pd
import numpy as np
import time
import os
import datetime as dt
from datetime import date, timedelta, datetime

inicio = time.time()

mst_faciales = pd.read_excel(rf'data.xlsx')
cons_valoracion = pd.read_excel(rf'tasas_dia.xlsx')

fechas = list((set(list(cons_valoracion["Fecha"]))))
fechas = sorted(fechas)

tasas_imputadas = pd.DataFrame(columns=["Fecha", "Instrumento", "Tasa"])

for i in fechas: #Ciclo i 
    tasas_fecha = cons_valoracion[cons_valoracion["Fecha"] == i]
    
    vigentes = mst_faciales[mst_faciales["Vencimiento"] > i]
    vigentes = vigentes[vigentes["Emisión"] < i]
    
    for nemo in vigentes["Nemo"]:

        if nemo in list(tasas_fecha["Instrumento"]):
            pass
        
        else:
            if i == fechas[0]:
                pass
            
            else:
                fecha_1 = fechas[fechas.index(i) - 1]
                tasas_fecha_1 = cons_valoracion[cons_valoracion["Fecha"] == fecha_1]
                
                if nemo in list(tasas_fecha_1["Instrumento"]):
                    
                    imputar = [i, nemo, tasas_fecha_1.iloc[list(tasas_fecha_1["Instrumento"]).index(nemo)][2]]
                
                    tasas_imputadas = tasas_imputadas.append(pd.DataFrame([imputar], columns = ["Fecha", "Instrumento", "Tasa"]), ignore_index = True)
                    cons_valoracion = cons_valoracion.append(pd.DataFrame([imputar], columns = ["Fecha", "Instrumento", "Tasa"]), ignore_index = True)

tasas_imputadas.to_excel('tasas_dia_n.xlsx')
