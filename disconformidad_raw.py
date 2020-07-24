# -*- coding: utf-8 -*-
from funciones.exploratorio import df, cod, categorizar_var, tabla_pond
import numpy as np
import pandas as pd

## Categorizamos las variables
l = from_to('ap1','ambito')
df[l] = categorizar_var(l)

#%%
##¿Creés que hay temas/actividades que la escuela debería abordar/enseñar y no lo hace? ap36
## Seleccionamos solo las categorías de interés
df['ap36_sel'] = df['ap36'].cat.set_categories(['1','2'])
# df['ap36'].cat.remove_categories(['1','2'])
nan_cat = ['-1','-6','-9']

## Sos... (sexo) ap2
df['ap2_sel'] = df['ap2'].cat.set_categories(['1','2'])

# Nivel educativo de la madre. ap10
cat = pd.Series(np.arange(1, 8).tolist(), dtype=str)
df['ap10_sel'] = df['ap10'].cat.set_categories(cat)

# ¿Qué vas a hacer cuando terminés el secundario?. ap10
df['ap47_sel'] = df['ap47'].cat.remove_categories(nan_cat)
#%%
mat = pd.get_dummies(df[['ap36_sel', 'ap2_sel', 'ap10_sel']], drop_first=True)
mat = pd.concat([df[['ap10_sel']].astype(float), pd.get_dummies(df[['ap36_sel', 'ap2_sel']], drop_first=True)], axis=1)
mat.dropna().corr()

sns.set()
import seaborn as sns

g = sns.catplot(x='ap36_sel_2', hue='ap10_sel', row='ap2_sel_2', kind='count', data=mat)

#%%
mat = df[['ap36_sel', 'ap2_sel', 'ap10_sel', 'ponder']]

tabla = (df.groupby(['ap36_sel', 'ap2_sel', 'ap10_sel'])['ponder']
    .sum()
    .reset_index())

tabla['rel_ap36'] = (tabla
 .groupby(['ap2_sel', 'ap10_sel'])['ponder']
 .transform(lambda x: x/sum(x)))

g = sns.catplot(y='rel_ap36', x='ap36_sel', hue='ap10_sel', row='ap2_sel', kind='bar', data=tabla)

import statsmodels.api as sm

## Realizamos un modelo lineal
X = pd.get_dummies(df[['ap2_sel', 'ap10_sel']], drop_first=True)
y = df['ap36_sel'].astype(float)

## Logistic model
logistic_model = sm.Logit(y, X)

#%%
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
import pandas as pd
import statsmodels.api as sm

y, X = dmatrices('ap36_sel ~ ap2_sel + ap10_sel + ambito + sector', df, return_type = 'dataframe')

model = LogisticRegression(fit_intercept = True)
mdl = model.fit(X, y.iloc[:,0])
model.coef_

# Output from statsmodels
logit = sm.Logit(y.iloc[:,0], X).fit()
logit.summary()
sm.stats.anova_lm(logit, typ=2)

#%%
df['mdesemp'] = df['mdesemp'].astype('float')
df['ldesemp'] = df['ldesemp'].astype('float')

g = sns.catplot(y='mdesemp', x='ap36_sel', hue='ap2_sel', col='ap10_sel', row='ambito', kind='bar', data=df)
sns.barplot(y='mdesemp', x='ap36_sel', data=df[['mdesemp', 'ap36_sel']].dropna())

g = sns.catplot(y='mdesemp', x='ap36_sel', hue='ap2_sel', col='ap10_sel', row='sector', kind='bar', data=df)
g = sns.catplot(y='ldesemp', x='ap36_sel', hue='ap2_sel', col='ap10_sel', row='sector', kind='bar', data=df)

g = sns.catplot(x='ap47_sel', hue='ap36_sel', kind='count', data=df)

#%% Decisión de estudiar en función de la disconformidad con los contenidos

tabla = (df.groupby(['ap47_sel', 'ap36_sel'])['ponder']
    .sum()
    .reset_index())

tabla['rel_ap47'] = (tabla
 .groupby(['ap47_sel'])['ponder']
 .transform(lambda x: x/sum(x)))

g = sns.catplot(y='rel_ap47', x='ap47_sel', hue='ap36_sel', kind='bar', data=tabla)


tabla = (df.groupby(['ap47_sel', 'ap36_sel', 'ap2_sel', 'ap10_sel'])['ponder']
    .sum()
    .reset_index())

tabla['rel_ap47'] = (tabla
 .groupby(['ap47_sel', 'ap2_sel', 'ap10_sel'])['ponder']
 .transform(lambda x: x/sum(x)))

g = sns.catplot(y='rel_ap47', x='ap47_sel', hue='ap36_sel', row='ap2_sel', col='ap10_sel', kind='bar', data=tabla)