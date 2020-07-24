# -*- coding: utf-8 -*-

def cod_sec_2017():
    import pandas as pd
    cod = pd.read_excel('./docs/dicc_secundaria.xlsx', index=[0,1], dtype='object')

    cod[['Variable', 'Etiqueta']] = cod[['Variable', 'Etiqueta']].ffill()
    cod = cod.dropna()
    cod = cod.set_index(['Variable'])
    cod = cod.apply(lambda x: x.astype(str))

    return cod



# cod = pd.read_excel('./docs/dicc_secundaria.xlsx', index=[0,1], dtype='object')

# cod[['Variable', 'Etiqueta']] = cod[['Variable', 'Etiqueta']].ffill()