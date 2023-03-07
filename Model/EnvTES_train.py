# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:28:49 2023

@author: David Mejia
"""

import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle



# Factor de normalización de negociación
# 1.000.000 de valor nominal por operación (cifras en miles)
TES_NORMALIZE = 1000
# Cantidad inicial de dinero en el balance disponible para operar (cifras en miles)
BALANCE_INICIAL_CUENTA = 10000000
# Número de títulos disponibles en el mercado
TES_DIM = None
# Cargos por transacción: 0.1% por tener cupo en SEN
CARGO_TRANSACCION = 0.001
# Costo anualizado de realización de simultáneas
TASA_SIMULTANEA = 0.1

REWARD_SCALING = 1e-4



class TESEnvEntr(gym.Env):
    """Entorno para negociación de TES en gym"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, day = 0):
        
        self.day = day
        self.df = df
        
        # Creación de espacios de acción y observación
        self.action_space =  spaces.Box(low = -1, high = 1, shape = (TES_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))
        
        # Cargar datos desde DataFrame
        self.data = self.df.loc[self.day,:]
        self.terminal = False
        
        # Inicializar el estado
        self.state = [BALANCE_INICIAL_CUENTA] + \
                      self.data.Tasa.values.tolist() + \
                      self.data["Precio Limpio"].values.tolist() + \
                      self.data["Precio Sucio"].values.tolist() + \
                      [0]*TES_DIM + \
                      self.data["Duración"].values.tolist() + \
                      self.data["Duración Modificada"].values.tolist() + \
                      self.data.Dv01.values.tolist() + \
                      self.data.Convexidad.values.tolist()
                      
        # Inicializar sistema de recompensas
        self.recompensa = 0
        self.costo = 0
        
        # Memorizar los cambios en el balance
        self.memoria_activos = [BALANCE_INICIAL_CUENTA]
        self.memoria_recompensa = []
        self.trades = 0
        #self.reset()
        self._seed()
    
