# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:43:44 2023

@author: David Mejia
"""
import pandas as pd
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt

from EnvTES_train import TESEnvEntr
from EnvTES_val import TESEnvVal
from EnvTES_trade import TESEnvTrade

import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv

import warnings
warnings.filterwarnings("ignore")


def train_A2C(env_train, model_name, timesteps=25000):
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"Working/{model_name}")
    print(' - Training time (A2C): ', (end - start) / 60, ' minutes')
    return model



def train_PPO(env_train, model_name, timesteps=50000):
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"Working/{model_name}")
    print(' - Training time (PPO): ', (end - start) / 60, ' minutes')
    return model



def train_DDPG(env_train, model_name, timesteps=10000):
    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"Working/{model_name}")
    print(' - Training time (DDPG): ', (end-start)/60,' minutes')
    return model



def encontrar_sharpe_validacion(iteration):
    # iteration = 756
    df_total_value = pd.read_csv('csv/validation/account_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()
             
    
    if np.isnan(sharpe):
        return 0         
    else:
        return sharpe



def DRL_validation(model, test_data, test_env, test_obs) -> None:
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)
        


def prediccion_DRL(df,
                   model,
                   name,
                   last_state,
                   iter_num,
                   unique_trade_date,
                   rebalance_window,
                   initial):

    trade_data = datasplit(df, inicio=unique_trade_date[iter_num - rebalance_window], final=unique_trade_date[iter_num])
    env_trade = DummyVecEnv([lambda: TESEnvTrade(trade_data,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteracion=iter_num)])
    obs_trade = env_trade.reset()

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('csv/trading/last_state_{}_{}.csv'.format(name, i), index=False)
    return last_state


def datasplit(df, inicio, final):
    
    # data = cons_TES
    data = df[(df.Fecha >= inicio) & (df.Fecha < final)]
    
    maestro_TES = pd.read_excel(r'data.xlsx')
    data["Vigente"] = 1
    new_data = pd.DataFrame(columns = data.columns) 
    aux_data = data.copy()
    for i in data.Fecha.unique():

        filtered_data = aux_data[aux_data["Fecha"] == i]
        
        for j in maestro_TES["Nemo"]:

            if j not in list(filtered_data["Instrumento"]):

                new_element = [i, j, float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0), float(0)]
                new_data.loc[len(new_data)] = new_element
        
            
    data = pd.concat([data, new_data])  
    
    
    data = data.sort_values(['Fecha','Instrumento'], ignore_index = True)
    data.index = data.Fecha.factorize()[0]
    
    return data
    
    



def estrategia_ensamblada(df, fechas_bursatiles, rebalanceo, validacion):
    # Parámetros 
    # df = cons_TES
    # fechas_bursatiles = unique_trade_date
    # rebalanceo = ventana_rebalanceo
    # validacion = ventana_val
    
    #run_complete()
    
    ult_estado_ens = []
    ult_estado_ddpg = []
    ult_estado_a2c = []
    ult_estado_ppo = [] 
    
    
    ppo_perform = []
    ddpg_perform = []
    a2c_perform = []
    
    uso_modelo = []
    
    inicio = time.time()
    for i in range(2*rebalanceo + 2*validacion, len(fechas_bursatiles), rebalanceo):
        # i = 2*rebalanceo + 2*validacion
        if i - 2*rebalanceo - 2*validacion == 0:
            # Se identifica que este es el estado inicial
            initial = True
        else:
            # No es el estado inicial
            initial = False
        
        # ides.append(initial)
        print("-" * 50)
        # Se separa dataset para entrenamiento utilizando función datasplit y se genera un entorno de entrenamiento
        entrenamiento = datasplit(df, fechas_bursatiles[0], fechas_bursatiles[i - rebalanceo - validacion])
        env_entr = DummyVecEnv([lambda: TESEnvEntr(entrenamiento)])
        
        # Se separa dataset para validación utilizando función datasplit y se genera un entorno de validación
        validate = datasplit(df, fechas_bursatiles[i - rebalanceo - validacion], fechas_bursatiles[i - rebalanceo])
        env_val = DummyVecEnv([lambda: TESEnvVal(validate, iteracion=i)])
        
        obs_val = env_val.reset()
        
        print(" - Entrenamiento del Modelo desde: ", fechas_bursatiles[0], "hasta: ",
              fechas_bursatiles[i - rebalanceo - validacion])
        print(" - Entrenamiento A2C")
        model_a2c = train_A2C(env_entr, model_name="A2C_FixedIncome_Col_{}".format(i), timesteps=30000)
        print(" - Validación A2C desde: ", fechas_bursatiles[i - rebalanceo - validacion], "to ",
              fechas_bursatiles[i - rebalanceo])
        DRL_validation(model=model_a2c, test_data=validate, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = encontrar_sharpe_validacion(i)
        print(" - A2C Sharpe Ratio: ", sharpe_a2c)
    
        print(" - Entrenamiento PPO")
        model_ppo = train_PPO(env_entr, model_name="PPO_FixedIncome_Col_{}".format(i), timesteps=100000)
        print(" - Validación PPO desde: ", fechas_bursatiles[i - rebalanceo - validacion], "hasta: ",
              fechas_bursatiles[i - rebalanceo])
        DRL_validation(model=model_ppo, test_data=validate, test_env=env_val, test_obs=obs_val)
        sharpe_ppo = encontrar_sharpe_validacion(i)
        print(" - PPO Sharpe Ratio: ", sharpe_ppo)
    
        print(" - Entrenamiento DDPG")
        model_ddpg = train_DDPG(env_entr, model_name="DDPG_FixedIncome_Col_{}".format(i), timesteps=10000)
        print(" - Validación DDPG desde: ", fechas_bursatiles[i - rebalanceo - validacion], "hasta: ",
              unique_trade_date[i - rebalanceo])
        DRL_validation(model=model_ddpg, test_data=validate, test_env=env_val, test_obs=obs_val)
        sharpe_ddpg = encontrar_sharpe_validacion(i)
        print(" - DDPG Sharpe Ratio: ", sharpe_ddpg)
    
        ppo_perform.append(sharpe_ppo)
        a2c_perform.append(sharpe_a2c)
        ddpg_perform.append(sharpe_ddpg)
        
        # Selección del Modelo Basado en Sharpe Ratio
        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
            model_ensemble = model_ppo
            uso_modelo.append('PPO')
        elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
            model_ensemble = model_a2c
            uso_modelo.append('A2C')
        else:
            model_ensemble = model_ddpg
            uso_modelo.append('DDPG')
    
        print(" - Trading desde: ", fechas_bursatiles[i - rebalanceo], "hasta ", fechas_bursatiles[i])
        print("-" * 50)
        ult_estado_ens = prediccion_DRL(df=df, model=model_ensemble, name="ensemble",
                                             last_state=ult_estado_ens, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalanceo,
                                             initial=initial)
        
        ult_estado_ppo = prediccion_DRL(df=df, model=model_ppo, name="ppo",
                                             last_state=ult_estado_ppo, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalanceo,
                                             initial=initial)
        
        ult_estado_a2c = prediccion_DRL(df=df, model=model_a2c, name="a2c",
                                             last_state=ult_estado_a2c, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalanceo,
                                             initial=initial)
        
        ult_estado_ddpg = prediccion_DRL(df=df, model=model_ddpg, name="ddpg",
                                             last_state=ult_estado_ddpg, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalanceo,
                                             initial=initial)
        
    final = time.time()
    print("Ensemble Strategy took: ", (inicio - final) / 60, " minutes")
    
    return uso_modelo, ppo_perform, a2c_perform, ddpg_perform
    


if __name__ == "__main__":
    
    #Importamos nuestra base de datos
    cons_TES = pd.read_excel(r'consolidado_total.xlsx')
        
    fechas_int = []
    for i in cons_TES.index:
        
        # i = 0
        fecha_int = int(cons_TES.iloc[i][0].strftime('%Y') + cons_TES.iloc[i][0].strftime('%m') + cons_TES.iloc[i][0].strftime('%d'))
        fechas_int.append(fecha_int)
        
    cons_TES["Fecha"] = fechas_int
    
    
    ventana_rebalanceo = 63
    ventana_val = 63
    
    unique_trade_date = cons_TES.Fecha.unique()
    # print(unique_trade_date)
    
    # Correr estrategia de ensamble
    uso_modelo, ppo_perform, a2c_perform, ddpg_perform = estrategia_ensamblada(df = cons_TES, 
                                                                                fechas_bursatiles = unique_trade_date,
                                                                                rebalanceo = ventana_rebalanceo,
                                                                                validacion = ventana_val)


