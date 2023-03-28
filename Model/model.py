# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:43:44 2023

@author: David Mejia
"""
import pandas as pd
import datetime as dt
import time

from EnvTES_train import TESEnvEntr
from EnvTES_val import TESEnvVal
from stable_baselines.common.vec_env import DummyVecEnv


def datasplit(df, inicio, final):
    
  
    data = df[(df.Fecha >= inicio) & (df.Fecha < final)]
    data = data.sort_values(['Fecha','Instrumento'], ignore_index = True)
    data.index = data.Fecha.factorize()[0]
    
    return data


if __name__ == "__main__":
    
    #Importamos nuestra base de datos
    cons_TES = pd.read_excel(r'consolidado_total.xlsx')
    
    fechas_int = []
    for i in cons_TES.index:
        
        # i = 0
        fecha_int = int(cons_TES.iloc[i][0].strftime('%Y') + cons_TES.iloc[i][0].strftime('%m') + cons_TES.iloc[i][0].strftime('%d'))
        fechas_int.append(fecha_int)
        
    cons_TES["Fecha"] = fechas_int
    
    
    ventana_rebalanceo = 21
    ventana_val = 21
    
    unique_trade_date = cons_TES.Fecha.unique()
    # print(unique_trade_date)
    

# Correr estrategia de ensamble

# inicio = time.time()
# Parámetros 
df = cons_TES
fechas_bursatiles = unique_trade_date
rebalanceo = ventana_rebalanceo
validacion = ventana_val

#run_complete()

ult_estado_ens = []
ppo_perform = []
ddpg_perform = []
a2c_perform = []

uso_modelo = []

inicio = time.time()
ides = []
for i in range(rebalanceo + validacion, len(fechas_bursatiles), rebalanceo):
    
    if i - rebalanceo - validacion == 0:
        # Se identifica que este es el estado inicial
        initial = True
    else:
        # No es el estado inicial
        initial = False
    
    # ides.append(initial)
    
    # Se separa dataset para entrenamiento utilizando función datasplit y se genera un entorno de entrenamiento
    entrenamiento = datasplit(df, fechas_bursatiles[0], fechas_bursatiles[i - rebalanceo - validacion])
    env_entr = DummyVecEnv([lambda: TESEnvEntr(entrenamiento)])
    
    # Se separa dataset para validación utilizando función datasplit y se genera un entorno de validación
    validate = datasplit(df, fechas_bursatiles[i - rebalanceo - validacion], fechas_bursatiles[i - rebalanceo])
    env_val = DummyVecEnv([lambda: TESEnvVal(validate, iteracion=i)])
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    