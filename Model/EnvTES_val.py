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
TES_DIM = 50
# Cargos por transacción: 0.1% por tener cupo en SEN
CARGO_TRANSACCION = 0.001
# Costo anualizado de realización de simultáneas
TASA_SIMULTANEA = 0.1

REWARD_SCALING = 1e-4



class TESEnvVal(gym.Env):
    """Entorno para negociación de TES en gym"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, day = 0, iteracion = ''):
        
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
                      self.data.DV01.values.tolist() + \
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
        
    def _buy_ticker(self, index, action):
        
        # perform buy action based on the sign of the action
        disponible = self.state[0] // self.state[index+1]
        # print('available_amount:{}'.format(available_amount))

        #update balance
        self.state[0] -= self.state[index + 1] * min(disponible, action)* \
                          (1+ CARGO_TRANSACCION)

        self.state[index + TES_DIM + 1] += min(disponible, action)

        self.cost += self.state[index + 1] * min(disponible, action)* \
                          CARGO_TRANSACCION
        self.trades += 1
     
        
    def _sell_ticker(self, index, action):
        
        # perform sell action based on the sign of the action
        if self.state[index + TES_DIM + 1] > 0:
            #update balance
            self.state[0] += \
            self.state[index+1] * min(abs(action),self.state[index + TES_DIM + 1]) * \
             (1- CARGO_TRANSACCION)

            self.state[index + TES_DIM + 1] -= min(abs(action), self.state[index + TES_DIM + 1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index + TES_DIM + 1]) * \
             CARGO_TRANSACCION
            self.trades+=1
        else:
            pass


    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('/images/validation/account_value_val.png')
            plt.close()
            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(TES_DIM+1)])*np.array(self.state[(TES_DIM+1):(TES_DIM*2+1)]))
            
            #print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('/csv/validation/account_value_val.csv')
            #print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
            #print("total_cost: ", self.cost)
            #print("total_trades: ", self.trades)
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(1)
            sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                  df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            #print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            #df_rewards.to_csv('/kaggle/working/account_rewards_train.csv')
            
            # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal,{}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * TES_NORMALIZE
            #actions = (actions.astype(int))
            
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(TES_DIM+1)])*np.array(self.state[(TES_DIM+1):(TES_DIM*2+1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))

            self.state = [self.state[0]] + \
                          self.data.Tasa.values.tolist() + \
                          self.data["Precio Limpio"].values.tolist() + \
                          self.data["Precio Sucio"].values.tolist() + \
                          list(self.state[(TES_DIM+1):(TES_DIM*2+1)]) + \
                          self.data["Duración"].values.tolist() + \
                          self.data["Duración Modificada"].values.tolist() + \
                          self.data.DV01.values.tolist() + \
                          self.data.Convexidad.values.tolist()

            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(TES_DIM+1)])*np.array(self.state[(TES_DIM+1):(TES_DIM*2+1)]))
            self.asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING



        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [BALANCE_INICIAL_CUENTA]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [BALANCE_INICIAL_CUENTA] + \
                      self.data.Tasa.values.tolist() + \
                      self.data["Precio Limpio"].values.tolist() + \
                      self.data["Precio Sucio"].values.tolist() + \
                      [0]*TES_DIM + \
                      self.data["Duración"].values.tolist() + \
                      self.data["Duración Modificada"].values.tolist() + \
                      self.data.DV01.values.tolist() + \
                      self.data.Convexidad.values.tolist()
        # iteration += 1 
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=1):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


