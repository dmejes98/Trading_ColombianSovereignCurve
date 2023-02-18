# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:43:44 2023

@author: David Mejia
"""


import pandas as pd
import datetime as dt
import time



if __name__ == "__main__":
    
    #Importamos nuestra base de datos
    cons_TES = pd.read_excel(r'consolidado_total.xlsx')
    
    fechas_int = []
    for i in cons_TES.index:
        
        # i = 0
        fecha_int = int(cons_TES.iloc[i][0].strftime('%Y') + cons_TES.iloc[i][0].strftime('%m') + cons_TES.iloc[i][0].strftime('%d'))
        fechas_int.append(fecha_int)
        
    cons_TES["Fecha"] = fechas_int
    
    
    ventana_train = 63
    ventana_val = 63
    
    unique_trade_date = cons_TES.Fecha.unique()
    # print(unique_trade_date)
    

# Correr estrategia de ensamble

# inicio = time.time()
# Parámetros 
# df = cons_TES


    