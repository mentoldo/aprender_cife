# -*- coding: utf-8 -*-
from funciones.exploratorio import (barras_apiladas, tabla_pond, from_to, df, cod, col_div, col_cat, barras_apiladas_setiq, etiquetas, val_lab,
                                    crear_paleta, parse_ylab, anotar_porcentajes, anotar_porcentaje_barras, barras_minimalista,
                                    col_cat_val, col_lab)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import textwrap
import matplotlib.ticker as mtick
import numpy as np

## Veamos las diferencias en cuanto a la demanda de otros conocimientos en 
## función del nivel educativo de los padres.

## ap11 nivel educativo del padre
cod.loc['ap11', ['Códigos', 'Etiqueta.1']]


df[df['ap11'].isin(np.arange(1,8).astype(str))]

#%%
## Cramos una tablas con las frecuencias con ponderaciones
t = df.groupby(['ap11', 'ap36'])['ponder'].sum().reset_index().pivot(index='ap36', columns='ap11')
## Tiramos el multiIndex
t.columns = t.columns.droplevel(0)
## Reordenamos para graficar
t=t.reindex(columns=t.columns.sort_values(ascending=False))

## Mapeamos los nombres de ylabel y legend
ylabel = val_lab('ap11')['Etiqueta.1'].to_dict()
legend = val_lab('ap36')['Etiqueta.1'].to_dict()

v=t.index.values
colores = col_cat_val(t.index.values)

#t.T.plot(kind='barh', stacked=True, colors=colores.values)
## Calculamos las frecuencias relativas ponderadas y seleccionamos los valores a graficar
tot=t.sum()
t = t.divide(tot, axis=1).loc[:,'8':'1']

## Graficamos
fig, ax = plt.subplots(figsize=(10,7))
t.T.plot(kind='barh',
         stacked=True,
         colors=colores.values,
         ax=ax)


ylabel = pd.Series(t.columns.map(ylabel))
ax.set(yticklabels = ylabel.apply(lambda x: textwrap.fill(x, width=25)),
       ylabel=col_lab(t.columns.name),
       xlim=(0,1))

ax.set_title(col_lab(t.index.name))

ax.legend(t.index.map(legend), loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False,
              shadow=False, ncol=len(legend))

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  

## Visualizamos las proporciones como porcentajes
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))

anotar_porcentajes(ax)

#%%


#t.reset_index

t.divide(tot, axis=1).loc['1', np.arange(1,8).astype(str).tolist()]
t.divide(tot, axis=1).loc['1', np.arange(1,8).astype(str).tolist()].index
t.divide(tot, axis=1).loc['1', np.arange(1,8).astype(str).tolist()].plot(kind='bar')
pd.DataFrame.stack