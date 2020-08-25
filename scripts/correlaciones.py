# -*- coding: utf-8 -*-
from funciones.exploratorio import df, categorizar_var, from_to
from src.abrir_etiquetas import abrir_etiquetas
import numpy as np
import itertools
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

et = abrir_etiquetas('docs/dicc_secundaria.xlsx')
l = from_to('ap1','ap49')

## Removemos las categorías nulas
nan_cat = ['-1','-6','-9']
df[l] = df[l].apply(lambda x: x.cat.remove_categories(nan_cat))
#%% Categorizamos las variables faltantes
l = from_to('ldesemp','iclimam')
df[l] = categorizar_var(l)
nan_cat = ['-1']

cat = df.isocioam.cat.categories.values.astype(int)
cat[cat < 0].astype(str)

def rem_cat(s):
    cat = s.cat.categories.values.astype(int)
    nan_cat = cat[cat < 0].astype(str)
    return s.cat.remove_categories(nan_cat)

df[l] = df[l].apply(rem_cat)
#%% Creamos una funcion auxiliar para calcular el coeficiente V de Cramer corregido
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def v_matrix(df):
    cols = df.columns.to_list()
    corrM = np.zeros((len(cols),len(cols)))

    for col1, col2 in itertools.combinations(cols, 2):
        idx1, idx2 = cols.index(col1), cols.index(col2)
        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]
    
    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    
    return corr

#%%
select = df.drop(['ponder', 'lpondera', 'mpondera',
                  'isocioal', 'isocioam',
                  'iclimal', 'iclimam',
                  'autoconmatematicam', 'autoconlengual',
                  'TEL', 'TEM', 'indicator'], axis=1)

corr = v_matrix(select)
np.fill_diagonal(corr.values, 1)

corr.to_csv('./reportes/correlaciones/v_cramer_matrix.csv')

# fig, ax = plt.subplots(figsize=(7, 6))
# ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
#%%
s = corr[['ap37f', 'ap37g', 'ap36']].sort_values(['ap36', 'ap37f', 'ap37g'])

