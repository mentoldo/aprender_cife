# -*- coding: utf-8 -*-
from src.preprocesing import prepr_code
from src.var_lab import Etiq

def abrir_etiquetas(file):
    '''
    Lee el archivo con las etiquetas y construye un objeto Etiq con la información de las etiquetas

    Parameters
    ----------
    file : str
        File path.

    Returns
    -------
    Etiq.

    '''
    cod = prepr_code(file)
    e = Etiq(cod)
    return e

# abrir_etiquetas('docs/dicc_secundaria.xlsx')