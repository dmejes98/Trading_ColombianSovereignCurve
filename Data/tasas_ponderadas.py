# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 19:10:00 2023

@author: David Mejia
"""

import pandas as pd
import numpy as np
import time
import os

libro = pd.read_excel(rf'SEN.xlsx')


inicio = time.time()
fechas = list(set(list(libro["Fecha"])))

consolidado_precios = pd.DataFrame(columns = ["Fecha", "Instrumento", "Tasa"])

for fecha in fechas:

    ops_dias = libro[libro["Fecha"] == fecha]
    
    nemos = list(set(list(ops_dias["Instrumento"])))
    
    for nemo in nemos:
        
        ops_nemos = ops_dias[ops_dias["Instrumento"] == nemo]
        
        ops_nemos["poderacion"] = ops_nemos["Giro"]/sum(ops_nemos["Giro"])
        
        tasa = round(sum(ops_nemos["poderacion"] * ops_nemos["Tasa"]), 3)
        lista = [fecha, nemo, tasa]
        consolidado_precios = consolidado_precios.append(pd.DataFrame([lista], columns = ["Fecha", "Instrumento", "Tasa"]), ignore_index = True)

final = time.time()

ejec_time = (final - inicio)/60
print("Tiempo de ejecución en minutos: ", ejec_time)

consolidado_precios.to_excel('tasas_dia.xlsx')
