# -*- coding: utf-8 -*-
from funciones.exploratorio import df, categorizar_var, from_to
from src.abrir_etiquetas import abrir_etiquetas
import numpy as np

et = abrir_etiquetas('docs/dicc_secundaria.xlsx')

l = from_to('ap1','ambito')
df[l] = categorizar_var(l)


dir(et)
et.var['lpondera'].var_name


df.describe()

#%% Primer agrupamiento. Variables sociodemográficas
sociodem = df.loc[:,'ap1':'ap8g'].join(df['TEL']).copy()

sociodem.get_dummies

sociodem = sociodem.astype('float64')
sociodem[sociodem > 0].isna().sum()


sociodem.get_dummies()

import seaborn as sns
sns.heatmap(sociodem.corr(method=), annot=True)
#%%
sociodem = df.loc[:,'ap1':'ap8g']

nan_cat = ['-1','-6','-9']
sociodem = sociodem.apply(lambda x: x.cat.remove_categories(nan_cat))

cor = pd.get_dummies(sociodem).corr()

cor = cor[cor > 0.3]
cor = cor.unstack()
cor = cor[cor < 1]
cor = cor.sort_values(ascending=False)

cor = cor.reset_index()

def extraer_var(s):
    return s.str.extract('(?P<var>.*)_')


cor['var_level_0'] = cor['level_0'].str.extract('(?P<var>.*)_')
cor['var_level_1'] = cor['level_1'].str.extract('(?P<var>.*)_')
cor['var_level_0'] == cor['var_level_1']

cor.level_0.str.extract('(?P<var>.*)_')


pd.DataFrame(cor, columns=['cor']).stack(level=-1)

(cor > 0.3).sum()
sns.heatmap(pd.get_dummies(sociodem).corr(), annot=True)
#%%
## Link: https://stackoverflow.com/questions/51859894/how-to-plot-a-cramer-s-v-heatmap-for-categorical-features

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


cols = ["Party", "Vote", "contrib"]
corrM = np.zeros((len(cols),len(cols)))
# there's probably a nice pandas way to do this
for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]

corr = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");

#%%
import itertools
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

sociodem = sociodem.dropna(how='all')

cols = sociodem.columns.to_list()
corrM = np.zeros((len(cols),len(cols)))
df = sociodem

for col1, col2 in itertools.combinations(cols, 2):
    idx1, idx2 = cols.index(col1), cols.index(col2)
    corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
    corrM[idx2, idx1] = corrM[idx1, idx2]
    
corr = pd.DataFrame(corrM, index=cols, columns=cols)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");

sns.clustermap(corr, annot=True)
ax.set_title("Cramer V Correlation between Variables")


#%%
sns.catplot(x='ap46c',
            y='TEM',
            kind='bar',
            data=df)

sns.catplot(x='isocioam',
            y='TEM',
            kind='bar',
            data=df)

sns.catplot(x='autoconmatematicam',
            y='TEM',
            kind='bar',
            data=df)

sns.catplot(x='cod_provincia',
            y='TEM',
            kind='bar',
            row='isocioa',
            data=df)


#%%
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


# cols = ["Party", "Vote", "contrib"]
# corrM = np.zeros((len(cols),len(cols)))
# # there's probably a nice pandas way to do this
# for col1, col2 in itertools.combinations(cols, 2):
#     idx1, idx2 = cols.index(col1), cols.index(col2)
#     corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
#     corrM[idx2, idx1] = corrM[idx1, idx2]

# corr = pd.DataFrame(corrM, index=cols, columns=cols)
# fig, ax = plt.subplots(figsize=(7, 6))
# ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");

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
sociodem = df.loc[:,'ap1':'ap8g']
sociodem = sociodem.dropna(how='all')
corr = v_matrix(sociodem)

antec_acad = df.loc[:,'ap9':'ap24']
antec_acad = antec_acad.dropna(how='all')
corr = v_matrix(antec_acad)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

clases_part = df.loc[:,'ap25':'ap30h']
clases_part = clases_part.dropna(how='all')
corr = v_matrix(clases_part)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

clases_part = df.loc[:,'ap31a':'ap32c']
clases_part = clases_part.dropna(how='all')
corr = v_matrix(clases_part)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

modalidades = df.loc[:,'ap33a':'ap34a']
modalidades = pd.get_dummies(modalidades, drop_first=True)
corr = modalidades.corr()
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

modalidades = df.loc[:,'ap35a':'ap35j']
corr = modalidades.astype(float).corr('spearman')
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

modalidades = df.loc[:,'ap37a':'ap37h']
corr = modalidades.astype(float).corr('spearman')
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)


modalidades = df.loc[:,'ap44a':'ap44f']
corr = modalidades.astype(float).corr('spearman')
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

modalidades = df.loc[:,'ap45a':'ap45i']
corr = modalidades.astype(float).corr('spearman')
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");
sns.clustermap(corr, annot=True)

#%%
select = df.drop(['ponder', 'lpondera', 'mpondera',
                  'isocioal', 'isocioam',
                  'autoconmatematicam', 'autoconlengual',
                  'TEL', 'TEM', 'indicator'], axis=1)

corr = v_matrix(select)
fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.heatmap(corr, annot=True, ax=ax); ax.set_title("Cramer V Correlation between Variables");

sns.clustermap(corr.drop('indicator').drop('indicator', axis=1).dropna(), annot=True)
#%%
sns.catplot(x='ap36',
            y='TEM',
            hue='ap2',
            row='ambito',
            col='sector',
            kind='bar',
            data=df)

from sklearn import preprocessing

scaler = preprocessing.
sel = df[['ap36', 'TEM']].dropna()
sel['TEM'] = sel['TEM'].transform(lambda x: (x - x.mean())/x.std())

sns.catplot(x='ap36',
            y='TEM',
            kind='bar',
            data=sel)

#%%
s = corr[['ap37f', 'ap37g', 'ap36']].sort_values(['ap36', 'ap37f', 'ap37g'])

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import hamming

sel = select.loc[:,'ap44a':'ap44f']
sel = sel.dropna()

sel = select.T
# sel = sel.sample(1000)
mergings = linkage(sel, method='weighted', metric='jaccard')
dendrogram(mergings,
           labels=sel.index,
           leaf_rotation=90,
           leaf_font_size=8,
           color_threshold=0.7)

mergings = linkage(sel, method='weighted', metric='hamming')
dendrogram(mergings,
           labels=sel.index,
           leaf_rotation=90,
           leaf_font_size=8,
           color_threshold=0.7)

sns.clustermap(mergings, annot=True)
