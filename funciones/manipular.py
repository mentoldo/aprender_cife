# -*- coding: utf-8 -*-
def categorizar_var(var, df, cod):
    '''Categoriza la variable. Convierte el string en CategoricalDtype. Devuelve las categorías y un diccionario con las equivalencias'''
    df[var] = df[var].astype(str)
    
    def cambiar_cat(s):
        from pandas.api.types import CategoricalDtype
        c = CategoricalDtype(cod.loc[s.name, 'Códigos'], ordered=True)
        return s.astype(c)
    
    return df[var].apply(cambiar_cat)

def from_to(str1, str2, cols):
    import numpy as np
    return cols[np.where(cols == str1)[0][0]:np.where(cols == str2)[0][0]+1]

def etiquetas(var, cod):
    from pandas.api.types import CategoricalDtype
    ref = cod.loc[var,['Códigos', 'Etiqueta.1']].reset_index(drop=True)
    
    ## Construimos las categorías
    c = CategoricalDtype(ref['Códigos'], ordered=True)
    
    ## Convertimos Códigos a categoría
    ref['Códigos'] = ref['Códigos'].astype(c)
    
    return ref.set_index('Códigos')

def label(var, cod):
    return cod.loc[var,'Etiqueta'].unique()[0]
