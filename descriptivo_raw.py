# -*- coding: utf-8 -*-
import pandas as pd
f = open('./data/aprender2017-secundaria-12.csv')
print(f.readlines(5))
f.close()

df = pd.read_csv('./data/aprender2017-secundaria-12.csv', sep='\t')

cod = pd.read_excel('./docs/dicc_secundaria.xlsx', index=[0,1], dtype='object')


cod[['Variable', 'Etiqueta']] = cod[['Variable', 'Etiqueta']].ffill()
cod = cod.dropna()
cod = cod.set_index(['Variable'])
cod = cod.apply(lambda x: x.astype(str))


cols = df.columns.values

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

df['ap1'] = df.ap1.astype(str)
eq = dict(zip(cod.loc[df.ap1.name, 'Códigos'], cod.loc[df.ap1.name, 'Etiqueta.1']))
c = CategoricalDtype(cod.loc[df.ap1.name, 'Códigos'], ordered=True)

df.ap1.astype(c)
filter
g = sns.catplot(y='ap1', data=df, kind='count',
            order=cod.loc[df.ap1.name, 'Códigos'],
            color='steelblue')
g.set_yticklabels(cod.loc[df.ap1.name, 'Etiqueta.1'])
g.set(ylabel=cod.loc[df.ap1.name, 'Etiqueta'].values[0],
      xlabel='Total de estudiantes')

cod['Códigos']

#%%
g = sns.catplot(y='ap1', data=df, kind='count',
            order=cod.loc[df.ap1.name, 'Códigos'],
            color='steelblue',
            col='ap2')
g.set_yticklabels(cod.loc[df.ap1.name, 'Etiqueta.1'])
g.set(ylabel=cod.loc[df.ap1.name, 'Etiqueta'].values[0],
      xlabel='Total de estudiantes')

#%%
df['ap1'] = df.ap1.astype(c)
df.ap1.value_counts(sort=False, normalize=True).sort_index(ascending=False).plot(kind='barh')

pd.Series.value_counts


tabla = df.ap1.value_counts(sort=False, normalize=True).sort_index(ascending=False)
sns.barplot(tabla,
            tabla.index,
            color='steelblue')
sns.despine()
#%%
def categorizar(var, df=df.copy()):
    '''Categoriza la variable. Convierte el string en CategoricalDtype. Devuelve las categorías y un diccionario con las equivalencias'''
    df[var] = df[var].astype(str)
    eq = dict(zip(cod.loc[df[var].name, 'Códigos'], cod.loc[df[var].name, 'Etiqueta.1']))
    c = CategoricalDtype(cod.loc[df[var].name, 'Códigos'], ordered=True)
    df[var] = df[var].astype(c)
    return (df, c, eq)

df, _, _ = categorizar('ap2')

fig, ax = plt.subplots(figsize=(10,6))
def frecuencias(var, ax):
    tabla = df[var].value_counts(sort=False, normalize=True).sort_index(ascending=False)
    sns.barplot(tabla,
                tabla.index,
                color='steelblue',
                ax=ax)
    sns.despine()
    ax.set_yticklabels(cod.loc[df[var].name, 'Etiqueta.1'])
    ax.set(ylabel=cod.loc[df[var].name, 'Etiqueta'].values[0],
      xlabel='Proporción de estudiantes')
    plt.tight_layout()
    plt.show()
    
frecuencias('ap1', ax)
var='ap2'
frecuencias('ap2', ax)

fig, ax = plt.subplots(figsize=(10,6))
categorizar('ap2')
frecuencias('ap2', ax)

fig, ax = plt.subplots(figsize=(10,6))
df, _, _ = categorizar('ap3a')
frecuencias('ap3a', ax)
#%% Definimos otro categorizar
def categorizar_var(var, df=df.copy()):
    '''Categoriza la variable. Convierte el string en CategoricalDtype. Devuelve las categorías y un diccionario con las equivalencias'''
    df[var] = df[var].astype(str)
    
    def cambiar_cat(s):
        c = CategoricalDtype(cod.loc[s.name, 'Códigos'], ordered=True)
        return s.astype(c)
    
    return df[var].apply(cambiar_cat)

def from_to(str1, str2, lista=cols):
    import numpy as np
    return cols[np.where(cols == str1)[0][0]:np.where(cols == str2)[0][0]+1]

l = from_to('ap5a','ap5f')
df[l] = categorizar_var(l)

#%%
for var in l:
    fig, ax = plt.subplots(figsize=(10,6))
    frecuencias(var, ax)

var='ap5a'


(df[l] == '1').sum() / (~df[l].isna()).sum()
df[l].apply(pd.Series.value_counts) / (~df[l].isna()).sum()

(df[l].apply(pd.Series.value_counts) / (~df[l].isna()).sum()).T.plot(kind='bar', stacked = True)
pd.crosstab(df[l])

tabla = df[var].value_counts(sort=False, normalize=True).sort_index(ascending=False)
sns.barplot(tabla,
            tabla.index,
            color='steelblue',
            ax=ax)
sns.despine()
ax.set_yticklabels(cod.loc[df[var].name, 'Etiqueta.1'])
ax.set(ylabel=cod.loc[df[var].name, 'Etiqueta'].values[0],
  xlabel='Proporción de estudiantes')
plt.tight_layout()
plt.show()

cols = df.columns.values.tolist()
cols = cols[cols.index('ap5a'): cols.index('ap5f')+1]



d = pd.Series(from_to('ap5a','ap5f')).apply(lambda x: categorizar_var(x)).T

apply(categorizar_var)
categorizar_var('ap1')