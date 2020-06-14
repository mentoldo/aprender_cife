# -*- coding: utf-8 -*-
def categorizar_var(var, df, cod):
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

def to_float(var, df):
    return df[var].str.replace(',', '.').astype('float64')

def from_to(str1, str2, cols):
    import numpy as np
    return cols[np.where(cols == str1)[0][0]:np.where(cols == str2)[0][0]+1]

def val_lab(var, cod):
    from pandas.api.types import CategoricalDtype
    ref = cod.loc[var,['Códigos', 'Etiqueta.1']].reset_index(drop=True)
    
    ## Construimos las categorías
    c = CategoricalDtype(ref['Códigos'], ordered=True)
    
    ## Convertimos Códigos a categoría
    ref['Códigos'] = ref['Códigos'].astype(c)
    
    return ref.set_index('Códigos')

def col_lab(var, cod):
    return cod.loc[var,'Etiqueta'].unique()[0]

def tabla_pond(var, df, rel=True):
    '''Toma una lista de nombres de variables y devuelve una tabla de frecuencias.
    var: list of strings
    rel: True - False. Define si se calculan las frecuencias relativas o absolutas.
    '''
    
    tabla = df[var].apply(lambda x: df.groupby(x)['ponder'].sum())
    
    if rel:
        tabla = tabla / tabla.sum()
    
    return tabla

def tabla(var, df, rel=True):
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
