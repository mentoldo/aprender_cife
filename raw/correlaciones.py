# -*- coding: utf-8 -*-
from funciones.exploratorio import df, categorizar_var, from_to
from src.abrir_etiquetas import abrir_etiquetas

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