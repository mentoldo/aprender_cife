# -*- coding: utf-8 -*-
from funciones.exploratorio import barras_apiladas, tabla_pond, from_to, df, cod, col_div, barras_apiladas_setiq, etiquetas, crear_paleta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

l = from_to('ap35a', 'ap35j')
l = l[::-1]
t=tabla_pond(l)

color = crear_paleta(t, 'Blues', True).values

## Creamos los nombres de las categorías
import textwrap
import re

colnames, valnames = etiquetas(t)
valnames.reindex(t.index)


exp = re.compile('(?P<title>.*[…?])(?P<label>.*)')
colnames = colnames.str.extract(exp)

# Agregamos breaklines para la visualización
colnames['label'] = colnames['label'].apply(lambda x: textwrap.fill(x, width=30))
##########################################

t.columns = colnames['label']
## Creamos un DF apilado para pasar a sns
stacked = t.stack().reset_index()
stacked.columns = ['val_code', 'var', 'f']

stacked.val_code.cat.categories = valnames['Etiqueta.1'].values

#sns.palplot(color)
sns.set_palette(color)
fig, ax = plt.subplots(figsize=(10,6))
g = sns.barplot(y='var', x='f', hue='val_code', data=stacked, ax=ax)
g.set(ylabel='')

#ax.legend('')
#
#g._l
#
#ax.set_yticklabels(colnames.label)

ax.legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=False, shadow=False, ncol=len(valnames['Etiqueta.1']),
          facecolor='inherit')
#%%
## 2)) ### Con catplot ############
####################################
l = from_to('ap35a', 'ap35j')
l = l[::-1]
t=tabla_pond(l)

color = crear_paleta(t, 'Blues', True).values

## Creamos los nombres de las categorías
colnames, valnames = etiquetas(t)
valnames.reindex(t.index)


exp = re.compile('(?P<title>.*[…?])(?P<label>.*)')
colnames = colnames.str.extract(exp)

# Agregamos breaklines para la visualización
colnames['label'] = colnames['label'].apply(lambda x: textwrap.fill(x, width=30))
##########################################

t.columns = colnames['label']
## Creamos un DF apilado para pasar a sns
stacked = t.stack().reset_index()
stacked.columns = ['val_code', 'var', 'f']

stacked.val_code.cat.categories = valnames['Etiqueta.1'].values

sns.set_palette(color)
#fig, ax = plt.subplots(figsize=(10,6))
g = sns.catplot(y='var', x='f', hue='val_code', data=stacked, kind='bar', legend=False, height=10)
g.ax.set(ylabel='', xlabel='Proporción de estudiantes')
g.ax.set_xlim((0,1))
g.ax.legend(loc='lower center', ncol=len(valnames['Etiqueta.1']), bbox_to_anchor=(0.5, -0.15))
#g.fig.legend(loc='lower center', ncol=len(valnames['Etiqueta.1']), bbox_to_anchor=(0.5, -0.1))
plt.subplots_adjust(bottom=0.125)
plt.tight_layout()


#g.ax.legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=False, shadow=False, ncol=len(valnames['Etiqueta.1']))

#dir(g)