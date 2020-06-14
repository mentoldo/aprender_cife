# -*- coding: utf-8 -*-
from funciones.exploratorio import barras_apiladas, tabla_pond, from_to, df, cod
import matplotlib.pyplot as plt
import pandas as pd


l = from_to('ap35a', 'ap35j')
l = l[::-1]
t=tabla_pond(l)
fig, ax = plt.subplots(figsize=(12,10))
barras_apiladas(t, ax, parse_labels=True)


l = from_to('ap37a', 'ap37h')
l = l[::-1]
t=tabla_pond(l)
fig, ax = plt.subplots(figsize=(10,6))
barras_apiladas(t, ax)


l = from_to('ap37a', 'ap37h')
l = l[::-1]

dfl = list(df.groupby('ap2')[l])
fig, axs = plt.subplots(len(dfl),1, sharex=True, figsize=(10,8))
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas(t, axs[i])

dfl[0][1]
df.groupby('ap2').__iter__()



dfl = list(df.groupby('sector')[l])
fig, axs = plt.subplots(len(dfl),1, sharex=True, figsize=(10,8))
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas(t, axs[i])
    
import seaborn as sns

g = sns.FacetGrid(df, col='sector')
g.map(barras_apiladas)

sns.catplot

dfl = list(df.groupby('cod_provincia')[l])
fig, axs = plt.subplots(5,5, sharex=True, figsize=(10,8))
le = len(dfl)
for i in range(le):
    c = i//5
    f = i % 5
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas(t, axs[c,f])
    
