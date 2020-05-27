# -*- coding: utf-8 -*-
import pandas as pd
from funciones.abrir_bases import cod_sec_2017
from funciones.manipular import *

df = pd.read_csv('./data/aprender2017-secundaria-12.csv', sep='\t')
cod = cod_sec_2017()
cols = df.columns

#%% Funciones
categorizar_var(['ap1'], df=df, cod=cod)
def categorizar(var): return categorizar_var(var, df=df, cod=cod)


categorizar(['ap1'])