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
from pandas.api.types import CategoricalDtype
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
    from pandas.api.types import CategoricalDtype
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
df[['ap1']] = categorizar_var(['ap1'])

#%%
def frecuencias(var, ax):
    ## Construimos la tabla de distribución de frecuencias
    tabla = df[var].value_counts(sort=False, normalize=True)
#    col = tabla.index.values.map(lambda x: int(x) < 0).map({False:'steelblue', True:'grey'})
    
    ## Etiquetas y colores
    etiq = cod.loc[df[var].name, ['Códigos', 'Etiqueta.1']]
    etiq['color'] = etiq['Códigos'].map(lambda x: int(x) < 0).map({False:'steelblue', True:'grey'})
    etiq= etiq.set_index('Códigos').reset_index(drop=True)
    
    tabla = pd.concat([tabla.reset_index(), etiq], axis=1).set_index('index').sort_index(ascending=False)
    
#    tabla.index = tabla.index.rename_categories(etiq['Etiqueta.1'].values)
    
    
    ## Graficamos
    tabla[var].plot(kind='barh', width=0.8, ax=ax, color=tabla.color)
    
    ax.set_yticklabels(tabla['Etiqueta.1'])
    ax.set(ylabel=cod.loc[df[var].name, 'Etiqueta'].values[0],
      xlabel='Proporción de estudiantes',
      xlim=(0,1))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

fig, ax = plt.subplots(figsize=(10,6))
frecuencias('ap1', ax)
var='ap1'
#%%

for var in l:
    fig, ax = plt.subplots(figsize=(10,6))
    frecuencias(var, ax)

fig, axs = plt.subplots(3,1, sharex=True, figsize=(10,8))
for i in range(len(l)):
    frecuencias(l[i], axs[i])
plt.subplots_adjust(hspace=0.1)

var='ap5a'

l = from_to('ap3a','ap3c')
df[l] = categorizar_var(l)

(df[l] == '1').sum() / (~df[l].isna()).sum()
df[l].apply(pd.Series.value_counts) / (~df[l].isna()).sum()

(df[l].apply(pd.Series.value_counts) / (~df[l].isna()).sum()).plot(kind='bar', stacked = False, width=0.8)
pd.crosstab(df[l])



def etiquetas(var):
    from pandas.api.types import CategoricalDtype
    ref = cod.loc[var,['Códigos', 'Etiqueta.1']].reset_index(drop=True)
    
    ## Construimos las categorías
    c = CategoricalDtype(ref['Códigos'], ordered=True)
    
    ## Convertimos Códigos a categoría
    ref['Códigos'] = ref['Códigos'].astype(c)
    
    return ref.set_index('Códigos')

def label(var):
    return cod.loc[var,'Etiqueta'].unique()[0]

ref = etiquetas(var)
ref['Códigos']


l = from_to('ap3a','ap3c')
df[l] = categorizar_var(l)
tabla = df[l].apply(pd.Series.value_counts, normalize=True)

n = tabla.columns.map(label)
tabla = pd.concat([tabla, etiquetas('ap3a')], axis=1)

fig, ax = plt.subplots()
tabla[l].plot(kind='bar',
     label=n.values,
     ax=ax)

ax.legend(n)
ax.set_xticklabels(tabla['Etiqueta.1'])


#%% 
df[['ap4']] = categorizar_var(['ap4'])
fig, ax = plt.subplots()
frecuencias('ap4', ax=ax)

#%%
l = from_to('ap5a','ap5h')
df[l] = categorizar_var(l)
tabla = df[l].apply(pd.Series.value_counts, normalize=True)

n = tabla.columns.map(label)
tabla = pd.concat([tabla, etiquetas('ap5a')], axis=1)

fig, ax = plt.subplots()
tabla[l].T.plot(kind='barh',
     stacked=True,
     ax=ax,
     color=color)

ax.legend(tabla['Etiqueta.1'])
ax.set_yticklabels(n)
#%%
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
qual = cm.get_cmap('tab10', 10)

col(range(2))

gray = cm.get_cmap('Greys', 10)

gray([2,5,8])

color = np.vstack([col(range(2)), gray([2,5,8])])

#%%
df[['ap6']] = categorizar_var(['ap6'])
fig, ax = plt.subplots()
frecuencias('ap6', ax=ax)

#%%
l = from_to('ap7a','ap7d')
df[l] = categorizar_var(l)
tabla = df[l].apply(pd.Series.value_counts, normalize=True)

n = tabla.columns.map(label)
tabla = pd.concat([tabla, etiquetas('ap7a')], axis=1)

fig, ax = plt.subplots()
tabla[l].T.plot(kind='barh',
     stacked=True,
     ax=ax,
     color=color)

ax.legend(tabla['Etiqueta.1'])
ax.set_yticklabels(n)