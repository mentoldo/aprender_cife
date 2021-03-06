# -*- coding: utf-8 -*-
from funciones.exploratorio import (barras_apiladas, tabla_pond, from_to, df, cod, col_div, col_cat, barras_apiladas_setiq, etiquetas, val_lab,
                                    crear_paleta, parse_ylab, anotar_porcentajes, anotar_porcentaje_barras, barras_minimalista)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import textwrap
import matplotlib.ticker as mtick

def graficar_barras_apiladas(desde, hasta, tipo, reverse=True, reverse_color=True):
    l = from_to(desde, hasta)
    if reverse:
        l = l[::-1]    
    t=tabla_pond(l)
    fig, ax = plt.subplots(figsize=(10,7))
    
    def f(x):
        return {'sino': col_cat,
                'div': lambda x: col_div(x, reverse_color),
                'grad': lambda x: crear_paleta(x, 'Blues', reverse_color)}[x]     
        
    barras_apiladas_setiq(t, ax, f(tipo)(t).values)
    
    colnames, valnames = etiquetas(t)
    #valnames = valnames.reindex(t.index)
    
    colnames = parse_ylab(colnames, True)
    ax.set_yticklabels(colnames.label)
    
    #fig.subplots_adjust(bottom=0.1)
    ax.legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.07),
                  shadow=False, ncol=len(valnames['Etiqueta.1']))
    
    ax.set_xlabel('Proporción de alumnos')
    
    fig.tight_layout()

graficar_barras_apiladas('ap5a', 'ap5h', 'sino')
graficar_barras_apiladas('ap15a', 'ap15c', 'grad')
graficar_barras_apiladas('ap30a', 'ap30h', 'div')

#%% Barras apiladas
l = from_to('ap35a', 'ap35j')
l = l[::-1]    
t=tabla_pond(l)
fig, ax = plt.subplots(figsize=(10,7))

color = crear_paleta(t, 'Blues', True).values
barras_apiladas_setiq(t, ax, color)

colnames, valnames = etiquetas(t)
#valnames = valnames.reindex(t.index)

colnames = parse_ylab(colnames, True)
ax.set_yticklabels(colnames.label)

#fig.subplots_adjust(bottom=0.1)
ax.legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False,
              shadow=False, ncol=len(valnames['Etiqueta.1']))

ax.set(title=colnames.title[0],
          ylabel='',
          xlabel='Proporción de estudiantes',
          xlim=(0,1))

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  

ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))

fig.tight_layout()

ax.patches[0].get_width()
#%% Realizamos anotaciones

anotar_porcentajes(ax)


#%% Barras apiladas. Para grillas con varias filas

l = from_to('ap35a', 'ap35j')
l = l[::-1]
dfl = list(df.groupby('ap2')[l])[0:2]
## Creamos la paleta de colores
color = crear_paleta(t, 'Blues', True).values

fig, axs = plt.subplots(len(dfl), 1, sharex=True, sharey=True, figsize=(10,8))
m = val_lab('ap2')['Etiqueta.1'].to_dict()
for i in range(len(dfl)):
    t=tabla_pond(l, df=dfl[i][1])
    barras_apiladas_setiq(t, axs[i], color)
    colnames, valnames = etiquetas(t)
    colnames = parse_ylab(colnames, True, 40)
    axs[i].set_yticklabels(colnames.label, fontdict = {'fontsize':8})
    axs[i].set_title(m[dfl[i][0]].strip(), fontdict = {'fontsize':8})
    # Hide the right and top spines
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)

axs.flatten()[-1].set_xlabel('Proporción de estuciantes')

fig.subplots_adjust(top=0.92, left=0.10, right=.98, bottom=0.05)



axs.flatten()[-1].legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.4, -0.125),
              shadow=False, ncol=len(valnames['Etiqueta.1']), fontsize='small', frameon=False)  


fig.suptitle(colnames.title[0])

fig.tight_layout()
fig.subplots_adjust(top=0.92)
#
#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')

#%%

l = from_to('ap35a', 'ap35j')
l = l[::-1]
dfl = list(df.groupby('cod_provincia')[l])
fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(10,8))

m = val_lab('cod_provincia')['Etiqueta.1'].to_dict()

for k in range(len(dfl)):
    j=k%5
    i=k//5
    t=tabla_pond(l, df=dfl[k][1])
    barras_apiladas_setiq(t, axs[i,j], color)
    colnames, valnames = etiquetas(t)
    colnames = parse_ylab(colnames, True, 55)
    axs[i,j].set_yticklabels(colnames.label, fontdict = {'fontsize':5})
    axs[i,j].set_title(m[dfl[k][0]].strip(), fontdict = {'fontsize':5})
    # Hide the right and top spines
    axs[i,j].spines['right'].set_visible(False)
    axs[i,j].spines['top'].set_visible(False)
    
fig.subplots_adjust(top=0.97, left=0.15, right=.98, bottom=0.05)
axs.flatten()[-3].legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.125),
              shadow=False, ncol=len(valnames['Etiqueta.1']))   

axs[4,4].axis('off')

t = []
for k in range(len(dfl)):
    t.append(tabla_pond(l, df=dfl[k][1]))
    print(dfl[k])
for i in range(len(dfl)):
    print(i%5, i//5)


#%%
l = ['ap2'] 
t=tabla_pond(l)
t = t.sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(10,6))
colnames, valnames = etiquetas(t)
valnames = valnames.reindex(t.index)

## Construimos los colores
valnames['color'] = valnames.index.map(lambda x: int(x) < 0).map({False:'steelblue', True:'grey'})   

## Graficamos
t.iloc[:,0].plot(kind='barh',
                 width=0.8,
                 ax=ax,
                 legend=False,
                 color=valnames.color.values)

## Colocamos las etiquetas
ax.set_yticklabels(valnames['Etiqueta.1'])
ax.set_xticklabels('')

ax.set(title=colnames[0],
  ylabel='',
  xlabel='Proporción de estudiantes',
  xlim=(0,1))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xticks([])
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',
#    bottom=False)
#
#ax.axis


plt.tight_layout()
plt.show()

anotar_porcentaje_barras(ax)
#for p in ax.patches:
#    width, height = p.get_width(), p.get_height()
#    x, y = p.get_xy()
#    if width < 0.01:
#        continue     
#    size = 10
#    rot = 'horizontal'
#    ax.text(x+width + 0.04, 
#            y+height/2, 
#            '{:.2f}%'.format(width*100), 
#            horizontalalignment='center', 
#            verticalalignment='center',
#            fontsize=size,
#            rotation=rot)

#%%
l = ['ap2'] 
t=tabla_pond(l)
fig, ax = plt.subplots(figsize=(10,5))
barras_minimalista(t, ax)

#%%
sns.FacetGrid.map

pd.DataFrame.stack
pd.DataFrame.pivot
pd.DataFrame.unstack

sns.color_palette('Blues', 12)
