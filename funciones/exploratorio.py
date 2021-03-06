# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from funciones.abrir_bases import cod_sec_2017
import numpy as np


df = pd.read_csv('./data/aprender2017-secundaria-12.csv', sep='\t', encoding='iso-8859-3', na_values=' ', dtype='object')

## Convertimos algunas variables a numéricas
l = ['ponder', 'lpondera', 'mpondera', 'TEL', 'TEM']
df[l] = df[l].apply(lambda x: x.str.replace(',', '.').astype('float64'))

## Agregamos una variable indicadora
df['indicator'] = 1

cod = cod_sec_2017()
cols = df.columns

#%% Funciones
def categorizar_var(var, df=df.copy(), cod=cod):
    '''Categoriza la variable. Convierte el string en CategoricalDtype. Devuelve las categorías y un diccionario con las equivalencias
    Args:
        var: String. Nombre de la variable a categorizar
        df: Pandas DataFrame con los datos
        cod: Pandas DataFrame del codebook
    
    Returns:
        Pandas Series con los nombres como valores y los códigos como índice.
    '''
    df[var] = df[var].astype(str)
    
    def cambiar_cat(s):
        from pandas.api.types import CategoricalDtype
        c = CategoricalDtype(cod.loc[s.name, 'Códigos'], ordered=True)
        return s.astype(c)
    
    return df[var].apply(cambiar_cat)

def to_float(var, df=df):
    return df[var].str.replace(',', '.').astype('float64')

def from_to(str1, str2, cols=cols):
    import numpy as np
    return cols[np.where(cols == str1)[0][0]:np.where(cols == str2)[0][0]+1]

def val_lab(var, cod=cod):
    from pandas.api.types import CategoricalDtype
    ref = cod.loc[var,['Códigos', 'Etiqueta.1']].reset_index(drop=True)
    
    ## Construimos las categorías
    c = CategoricalDtype(ref['Códigos'], ordered=True)
    
    ## Convertimos Códigos a categoría
    ref['Códigos'] = ref['Códigos'].astype(c)
    
    return ref.set_index('Códigos')

def col_lab(var, cod=cod):
    return cod.loc[var,'Etiqueta'].unique()[0]

def tabla_pond(var, df=df, rel=True):
    '''Toma una lista de nombres de variables y devuelve una tabla de frecuencias.
    var: list of strings
    rel: True - False. Define si se calculan las frecuencias relativas o absolutas.
    '''
    
    tabla = df[var].apply(lambda x: df.groupby(x)['ponder'].sum())
    
    if rel:
        tabla = tabla / tabla.sum()
    
    return tabla

def tabla(var, df=df, rel=True):
    '''Toma una lista de nombres de variables y devuelve una tabla de frecuencias.
    Args:
        list of strings
    Result:
        True - False. Define si se calculan las frecuencias relativas o absolutas.
    '''
    
    tabla = df[var].apply(lambda x: df.groupby(x)['indicator'].sum())
    
    if rel:
        tabla = tabla / tabla.sum()
    
    return tabla

def etiquetas(t):
    ''' Toma una tabla y busca las etiquetas para los códigos de columna y de valores.
    Devuelve una tupla con 2 DataFrames (colnames, valnames)
    
    Args:
        Un DataFrame tabla() o tabla_pond()
        
    Returns:
        Una tupla con dos DataFrames (colnames, valnames)
    '''
    import pandas as pd
    colnames = pd.Series(t.columns, index=t.columns).apply(col_lab)
    valnames = val_lab(t.columns[0])
    return colnames, valnames
#%%
l = from_to('ap1','ambito')
df[l] = categorizar_var(l)

#%% Funciones Gráficas
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

def barras_minimalista(t, ax):
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
    ax.set_xticklabels('')
    
    ax.set(title=colnames[0],
      ylabel='',
      xlabel='Porcentaje de estudiantes',
      xlim=(0,1))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False) 
    ax.set_xticks([])
   
    anotar_porcentaje_barras(ax)

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

def col_cat_val(v):
    '''Construye una paletta de colores para graficar las tablas.
    t: tabla
    
    # Values
    Devuelve un DataFrame con un color para cada fila.'''
    from matplotlib import cm
    
    ## Identificamos las categorías menores a cero
    i = pd.Series(v.map(lambda x: int(x) < 0).values, index = v, dtype='bool')
    
    ## Evaluamos la cantidad de categorías por color
    n_cat = (~i).sum()
    #n_grey = i.sum()
    
    ## Seleccionamos los colores de las categorías color
    cat = cm.get_cmap('tab10', 10)
    
    color = pd.DataFrame(cat(range(n_cat)),index=v[~i])
    
    ## Seleccionamos los colores de las categorías gris
    gray = cm.get_cmap('Greys', 10)
    
    color = color.append(pd.DataFrame(gray([2,5,8]), index=v[i]))
    
    return color

def col_div(t, invertir=True):
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
    
    ## Creamos un np array con los colores
    if invertir:
        color = cat(np.linspace(3, 27, n_cat).astype(int))[::-1,:]
    else:
        color = cat(np.linspace(3, 27, n_cat).astype(int))
    
    color = pd.DataFrame(color,index=t.index[~i])
    
    ## Seleccionamos los colores de las categorías gris
    gray = cm.get_cmap('Greys', 10)
    
    color = color.append(pd.DataFrame(gray([2,5,8]), index=t.index[i]))
    
    return color

def crear_paleta(t, paleta, invertir=False):
    '''Construye una paleta 'paleta' colocando grises en los valores negativos.
    Args:    
        t: tabla
        paleta: Str. Un string correspondiente a los colormaps de Matplotlib
    
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
    cat = cm.get_cmap(paleta, 30)
    
    ## Creamos un np array con los colores
    if invertir:
        color = cat(np.linspace(10, 30, n_cat).astype(int))[::-1,:]
    else:
        color = cat(np.linspace(10, 30, n_cat).astype(int))
    
    color = pd.DataFrame(color,index=t.index[~i])
    
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
    
    plt.tight_layout()
    plt.show()
    

def barras_apiladas_setiq(t, ax, color):
    t.T.plot(kind='barh',
             stacked=True,
             ax=ax,
             color=color,
             legend=False)
    
def parse_ylab(colnames, parse=True, width=30):
    import re
    import textwrap
    
    if parse:
        exp = re.compile('(?P<title>.*[…?])(?P<label>.*)')
        colnames = colnames.str.extract(exp)
    else:
        colnames = pd.DataFrame(colnames, columns=['label'])
        colnames['title'] = ''
    # Agregamos breaklines para la visualización
    colnames['label'] = colnames['label'].apply(lambda x: textwrap.fill(x, width=width))
    return colnames

    
def anotar_porcentajes(ax):
#    i = 1
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if width < 0.01:
            continue     
        if width < 0.03:
            size = 'x-small'
            rot = 45
            pos = y+ height+height*0.05
            horizontalalignment='left'
            verticalalignment='bottom'
            xpos=x+width/8
            
        else: 
            size = 'small'
            rot = 'horizontal'
            pos = y+ height+height*0.15
            horizontalalignment='center'
            verticalalignment='center'
            xpos=x+width/2
#       if width < 0.03:
#            i = 1 - i
#            pos = y+height +(1 - i)*(height*0.15)-i*(height+height*0.15)
#        else:
#            pos = y+ height+height*0.15
            
#        pos = y+ height+height*0.15
        ax.text(xpos, 
                pos, 
                '{:.2f}%'.format(width*100), 
                horizontalalignment=horizontalalignment, 
                verticalalignment=verticalalignment,
                fontsize=size,
                rotation=rot)
        
def anotar_porcentaje_barras(ax):
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if width < 0.01:
            continue     
        size = 10
        rot = 'horizontal'
        ax.text(x+width + 0.04, 
                y+height/2, 
                '{:.2f}%'.format(width*100), 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=size,
                rotation=rot)
        
