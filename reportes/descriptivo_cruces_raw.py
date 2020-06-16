# -*- coding: utf-8 -*-
from funciones.exploratorio import barras_apiladas, tabla_pond, from_to, df, cod, col_div, barras_apiladas_setiq, etiquetas, crear_paleta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


l = from_to('ap35a', 'ap35j')
l = l[::-1]
t=tabla_pond(l)

color = crear_paleta(t, 'Blues', True).values
#t.columns.name = 'vars'
stacked = t.stack().reset_index()
stacked.columns = ['val_code', 'var', 'f']


sns.palplot(color)
sns.set_palette(color)
ax = sns.catplot(y='var', x='f', hue='val_code', data=stacked, kind='bar')


t.pivot(columns='vars', values=l)

fig, ax = plt.subplots(figsize=(12,10))
barras_apiladas(t, ax, parse_labels=True)

import textwrap
import re

l = from_to('ap37a', 'ap37h')
l = from_to('ap35a', 'ap35j')
l = l[::-1]
t=tabla_pond(l)
t=t*100
fig, ax = plt.subplots(figsize=(10,6))

color = crear_paleta(t, 'Blues', True).values
barras_apiladas_setiq(t, ax, color)

colnames, valnames = etiquetas(t)
valnames.reindex(t.index)


exp = re.compile('(?P<title>.*[…?])(?P<label>.*)')
colnames = colnames.str.extract(exp)

# Agregamos breaklines para la visualización
colnames['label'] = colnames['label'].apply(lambda x: textwrap.fill(x, width=30))

ax.set_yticklabels(colnames.label)



fig, ax = plt.subplots(figsize=(10,6))
barras_apiladas_setiq(t, ax)
ax.set

dfl = list(df.groupby('ap2')[l])[0:2]
fig, axs = plt.subplots(len(dfl), 1, sharex=True, sharey=True, figsize=(10,8))
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas_setiq(t, axs[i])
 
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')



l = from_to('ap37a', 'ap37h')
l = l[::-1]

dfl = list(df.groupby('ap2')[l])[0:2]
fig, axs = plt.subplots(1,len(dfl), sharex=True, sharey=True, figsize=(10,8))
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas(t, axs[i])
    
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

dfl[0][1]
df.groupby('ap2').__iter__()


t = t.stack().reset_index()
t.columns = ['val_code', 'var', 'f']

sns.catplot(y='var', x='f', hue='val_code', data=t, kind='bar')
dfl = list(df.groupby('sector')[l])
fig, axs = plt.subplots(len(dfl),1, sharex=True, figsize=(10,8))
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas(t, axs[i])
    

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
    
sns.FacetGrid.map

pd.DataFrame.stack
pd.DataFrame.pivot
pd.DataFrame.unstack

sns.color_palette('Blues', 12)
