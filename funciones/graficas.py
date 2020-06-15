# -*- coding: utf-8 -*-
from funciones.manipular import etiquetas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def barras(t, ax):
    t = t.sort_index(ascending=False)
    
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
    ax.set(title=colnames[0],
      ylabel='',
      xlabel='Proporción de estudiantes',
      xlim=(0,1))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def col_cat(t):
    '''Construye una paletta de colores para graficar las tablas.
    t: tabla
    
    # Values
    Devuelve un DataFrame con un color para cada fila.'''
    from matplotlib import cm
    
    colnames, valnames = etiquetas(t)
    
    ## Identificamos las categorías menores a cero
    i = pd.Series(valnames.index.map(lambda x: int(x) < 0), dtype='bool')
    
    ## Evaluamos la cantidad de categorías por color
    n_cat = (~i).sum()
    #n_grey = i.sum()
    
    ## Seleccionamos los colores de las categorías color
    cat = cm.get_cmap('tab10', 10)
    
    color = pd.DataFrame(cat(range(n_cat)),index=t.index[~i])
    
    ## Seleccionamos los colores de las categorías gris
    gray = cm.get_cmap('Greys', 10)
    
    color = color.append(pd.DataFrame(gray([2,5,8]), index=t.index[i]))
    
    return color

def col_div(t):
    '''Construye una paleta divergente de colores para graficar las tablas.
    Args:    
        t: tabla
    
    Return:
        Devuelve un DataFrame con un color para cada fila.'''
    from matplotlib import cm
    #from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    
    colnames, valnames = etiquetas(t)
    
    ## Identificamos las categorías menores a cero
    i = pd.Series(valnames.index.map(lambda x: int(x) < 0), dtype='bool')
    
    ## Evaluamos la cantidad de categorías por color
    n_cat = (~i).sum()
    #n_grey = i.sum()
    
    ## Seleccionamos los colores de las categorías color
    cat = cm.get_cmap('RdBu', 30)
    
    color = pd.DataFrame(cat(np.linspace(3, 27, n_cat).astype(int)),index=t.index[~i])
    
    ## Seleccionamos los colores de las categorías gris
    gray = cm.get_cmap('Greys', 10)
    
    color = color.append(pd.DataFrame(gray([2,5,8]), index=t.index[i]))
    
    return color

def barras_apiladas(t, ax, parse_labels=True):
    import textwrap
    import re
    colnames, valnames = etiquetas(t)
    valnames = valnames.reindex(t.index)
    
    if parse_labels:    
        ## Utilizamos expresiones regulares para separar el título de las etiquetas
        exp = re.compile('(?P<title>.*[…?])(?P<label>.*)')
        colnames = colnames.str.extract(exp)
    else:
        colnames = pd.DataFrame(colnames, columns=['label'])
        colnames['title'] = ''
        
    # Agregamos breaklines para la visualización
    colnames['label'] = colnames['label'].apply(lambda x: textwrap.fill(x, width=30))
    
    t.T.plot(kind='barh',
             stacked=True,
             ax=ax,
             color=col_div(t).values)

    ## Colocamos las etiquetas
    ax.set_yticklabels(colnames.label)
    ax.set(title=colnames.title[0],
              ylabel='',
              xlabel='Proporción de estudiantes',
              xlim=(0,1))
    
#    ax.legend(valnames['Etiqueta.1'])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Put a legend below current axis
    ax.legend(valnames['Etiqueta.1'], loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=False, ncol=len(valnames['Etiqueta.1']))
    
    #plt.tight_layout()
    plt.show()
