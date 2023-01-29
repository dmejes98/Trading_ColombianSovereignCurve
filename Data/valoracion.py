# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 20:25:06 2023

@author: David Mejia
"""

import pandas as pd
import numpy as np
import time
import os
import datetime as dt
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import itertools

def valorar(cupon, fecha_emi, fecha_act, fecha_vto, ytm):
    
    #Variables
    f_valo = fecha_act
    f_emi = fecha_emi 
    f_ven = fecha_vto
    t_cupon = cupon / 100
    t_yield = ytm / 100
    base = 365
    nominal = 100
    f = 12
 
    bisiestos=['1992-02-29','1996-02-29','2000-02-29','2004-02-29','2008-02-29','2012-02-29','2016-02-29',
                          '2020-02-29','2024-02-29','2028-02-29','2032-02-29','2036-02-29','2040-02-29','2044-02-29',
                          '2048-02-29','2052-02-29','2056-02-29','2060-02-29','2064-02-29','2068-02-29','2072-02-29',
                          '2076-02-29','2080-02-29','2084-02-29','2088-02-29','2092-02-29','2096-02-29']
    bisiestos2 = []
    for i in bisiestos:
        bisiestos2.append(datetime.strptime(i, '%Y-%m-%d').date())
        
    # f_valo = dt.date(2015, 1, 5)
    # f_emi = dt.date(2008, 7, 24) 
    # f_ven = dt.date(2024, 7, 24)
    # t_cupon = 10 / 100
    # t_yield = 7.123 / 100
    # base = 365
    # nominal = 100
    # f = 12   
    
    ### Fechas de pago de cada cupón
    #valo=pd.DataFrame(columns=['f_cupon','cupon','d_desc','a_desc','FC','FC*t','t^2+t','FC * (t^2 + t)'])
    valo = pd.DataFrame(columns = ['f_cupon','cupon','d_desc','a_desc','FC','FC1','FC2','FC3'])
    
    i = 1
    while f_emi + relativedelta(months = f * i) <= f_ven:
        if f_emi + relativedelta(months = f * i) > f_ven:
            valo.at[i,'f_cupon'] = f_ven
        elif f_emi + relativedelta(months = f * i) >= f_valo:
            valo.at[i,'f_cupon'] = f_emi + relativedelta(months = f * i)
        i = i + 1
    valo = valo.reset_index()
    valo = valo.drop('index', axis=1)
    nper = len(list(valo['f_cupon']))
    ### Cantidad de cupónes/pagos, cupones
    
    n = len(valo)
    
    valo['cupon'] = 0.0
    for i in range(n-1):
        valo.at[i,'cupon'] = nominal * t_cupon
    valo.at[n-1,'cupon'] = nominal * t_cupon + nominal
    
    ### Años al vencimiento de cada cupón, flujo de caja, etc
    
    valo['d_desc'] = 0.0
    try:
        for i in range(len(valo)):
            valo.at[i,'d_desc'] = valo['f_cupon'][i] - f_valo
            valo.at[i,'d_desc'] = valo.at[i,'d_desc'].days
            valo.at[i,'d_desc']  = valo.at[i,'d_desc']-len(set(bisiestos2).intersection([f_valo + timedelta(days=x+1) for x in range(0, (valo.at[i,'f_cupon'] - f_valo).days)]))
            valo['a_desc'] = valo['d_desc']/base
            valo['FC'] = valo['cupon']/(1+t_yield)**valo['a_desc']
            valo['FC1'] = valo['FC']*valo['a_desc']
            valo['FC2'] = (valo['a_desc']**2)+valo['a_desc']
            valo["FC3"] = valo['FC']*valo['FC2']
            
            duracion=sum(valo['FC1'])/sum(valo['FC'])
            duracion_mod = duracion/(1+t_yield)
            convexidad = sum(valo['FC3'])*(1/(sum(valo['FC'])*(1+t_yield**2)))
            precio_s  = round(sum(valo['FC']), 4)
            
    except:
        valo['d_desc'] = (f_ven - f_valo).days - len(set(bisiestos2).intersection([f_valo + timedelta(days=x+1) for x in range(0, (f_ven - f_valo).days)]))
        valo['a_desc'] = valo['d_desc']/base
        valo['FC'] = valo['cupon']/(1+t_yield)**valo['a_desc']
        valo['FC1'] = valo['FC']*valo['a_desc']
        valo['FC2'] = (valo['a_desc']**2)+valo['a_desc']
        valo["FC3"] = valo['FC']*valo['FC2']
        
        duracion=sum(valo['FC1'])/sum(valo['FC'])
        duracion_mod = duracion/(1+t_yield)
        convexidad = sum(valo['FC3'])*(1/(sum(valo['FC'])*(1+t_yield**2)))
        precio_s  = round(sum(valo['FC']), 4)
        
    if f_valo.strftime('%m %d') == f_ven.strftime('%m %d'):
        precio_s = precio_s - cupon
        nper = nper - 1
        
    dv_01 = - 0.0001 * 1000000000 * duracion_mod
    precio_l = (t_cupon * 100 / (t_yield)) * (1 - (1 / (1 + (t_yield))**nper)) + 100 / ((1 + (t_yield))**nper)
    precio_l = round(precio_l, 4)
    
    return duracion, duracion_mod, convexidad, precio_s, precio_l, dv_01


inicio = time.time()

mst_faciales = pd.read_excel(rf'data.xlsx')
cons_valoracion = pd.read_excel(rf'tasas_dia_n.xlsx')

duracions = []
duracion_mods = []
convexidads = []
precio_ss = []
precio_ls = []
dv_01s = []

for i in cons_valoracion.index:
    
    cupon = mst_faciales.iloc[list(mst_faciales["Nemo"]).index(cons_valoracion.iloc[i][1])][1]
    emision = mst_faciales.iloc[list(mst_faciales["Nemo"]).index(cons_valoracion.iloc[i][1])][3]
    f_val = cons_valoracion.iloc[i][0]
    vencimiento = mst_faciales.iloc[list(mst_faciales["Nemo"]).index(cons_valoracion.iloc[i][1])][2]
    ytm = cons_valoracion.iloc[i][2]

    duracion, duracion_mod, convexidad, precio_s, precio_l, dv_01 = valorar(cupon, emision, f_val, vencimiento, ytm)
    
    duracions.append(duracion)
    duracion_mods.append(duracion_mod)
    convexidads.append(convexidad)
    precio_ss.append(precio_s)
    precio_ls.append(precio_l)
    dv_01s.append(dv_01)


cons_valoracion["Precio Limpio"] = precio_ls
cons_valoracion["Precio Sucio"] = precio_ss
cons_valoracion["Duración"] = duracions
cons_valoracion["Duración Modificada"] = duracion_mods
cons_valoracion["DV01"] = dv_01s
cons_valoracion["Convexidad"] = convexidads


final = time.time()

ejec_time = (final - inicio)/60
print("Tiempo de ejecución en minutos: ", ejec_time)


# cons_valoracion.to_excel('consolidado_n.xlsx')
