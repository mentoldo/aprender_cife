# -*- coding: utf-8 -*-
import pandas as pd

def prepr_code(file):
    '''
    Lee y preprocesa los códigos en formato .xlsx

    Parameters
    ----------
    file : str
        Path a códigos.

    Returns
    -------
    df : pd.DataFrame
        pd.DataFrame con los códigos limpios.

    '''  
    df = abrir_cod(file)
    df = change_col_names(df)
    df = fill_nans(df)
    df = clean_cod_by_group(df)
    return df

def abrir_cod(file):
    '''
    Abre los códigos de las variables en .xlsx

    Parameters
    ----------
    file : str
        Path al libro de códigos

    Returns
    -------
    cod : pd.DataFrame
        df con los códigos.
    '''  
    cod = pd.read_excel(file, index=[0,1], dtype='object')
    return cod

def change_col_names(df):
    '''
    Coloca los nombres de las columnas a df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los códigos de las variables.

    Returns
    -------
    df : pd.DataFrame con códigos
        DataFrame con los códigos de las variables con las etiquetas ['var_code', 'var_name', 'val_code', 'val_name'].

    '''
    df = df.copy()
    df.columns =['var_code', 'var_name', 'val_code', 'val_name']
    return df

def fill_nans(df):
    df = df.copy()
    df[['var_code', 'var_name']] = df[['var_code', 'var_name']].ffill()
    return df


def clean_cod_by_group(df):
    '''
    Aplica remove_nans_in_group(df) a cada uno de los grupos en df con los códigos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con df.columns == ['var_code', 'var_name', 'val_code', 'val_name']

    Returns
    -------
    df : pd.DataFrame
        pd.DataFrame con espacios en blanco de la columna val_code o var_code eliminados.

    '''
    df = df.copy()
    grupos = iter(df.groupby('var_code'))
    df = pd.concat(map(lambda x: remove_nans_in_group(x[1]), grupos)).sort_index().reset_index(drop=True)
    return df

def remove_nans_in_group(df):
    '''Remueve las filas con valores faltantes para df. df es una tabla con códigos con
    las siguientes columnas ['var_code', 'var_name', 'val_code', 'val_name']
    
    Args
        df(pdDataFrame): pd.DataFrame con df.columns == ['var_code', 'var_name', 'val_code', 'val_name']
    '''  
    ## Si la variable es numérica, retornamos el df
    if len(df) == 1:
        return df
    return df[~df[['val_code', 'val_name']].isna().any(axis=1)]
