# -*- coding: utf-8 -*-
from pandas.api.types import CategoricalDtype
import pandas as pd

class VarEtiq():
    '''
    Ordena información de etiquetado de una variable.
    
    Args:
        df (DataFrame): DataFrame con las columnas ['var_code', 'var_name', 'val_code', 'val_name']
    
    Attributes:
        var_name (str): Nombre de la variable
        val_cod (pd.Categorical): Códigos de la variable
        val_items (:obj:'list'): Lista de tuplas
    
    '''
    
    def __init__(self, df, ordered=True):
        assert not df.empty, 'El DataFrame está vacío.'
        assert list(df.columns) == ['var_code', 'var_name', 'val_code', 'val_name'],\
            "df.columns no es ['var_code', 'var_name', 'val_code', 'val_name']"
        
        self.var_code = df['var_code'].unique()[0]
        self.var_name = df['var_name'].unique()[0]
        self.ordered = ordered
        self.df = df[['val_code', 'val_name']]
        
        ## Determinamos si se trata de una variable etiquetada.
        if df.shape[0] == 1:
            self.labeled = False
        else:
            self.labeled = True
    
    def val_codes(self):
        if self.labeled:
            res = CategoricalDtype(categories=self.df.val_code, ordered=self.ordered)
        else:
            res = None
        return res
    
    def val_names(self):
        if self.labeled:
            res = CategoricalDtype(categories=self.df.val_name, ordered=self.ordered)
        else:
            res = None
        return res
#%%

class VarEtiq():
    '''
    Ordena información de etiquetado de una variable.    
    '''   
    def __init__(self, var_code, var_name, val_df, ordered=True):
        self.var_code = var_code
        self.var_name = var_name
        if val_df is not None:
            self.vals = ValEtiq(val_df, ordered)
        else:
            self.vals = None
            
    
    
class ValEtiq():
    '''Tiene información sobre las etiquetas de los valores de una variable'''
    def __init__(self, val_df, ordered=True):
        assert not val_df.empty, 'El DataFrame está vacío.'
        assert list(val_df.columns) == ['val_code', 'val_name'], "df.columns no es ['var_code', 'var_name']"
        
        self.val_df = val_df[['val_code', 'val_name']]
        self.ordered = ordered
    
    def val_codes(self):
        return CategoricalDtype(categories=self.val_df.val_code, ordered=self.ordered)
    
    def val_names(self):
        return CategoricalDtype(categories=self.val_df.val_name, ordered=self.ordered)
    
    def val_dict(self):
        return self.val_df.set_index('val_code')['val_name'].to_dict()
    
    


#%%
c = cod.loc['ap36'].reset_index()
c.columns = ['var_code', 'var_name', 'val_code', 'val_name']
val_df = c.reset_index()[['val_code', 'val_name']]


cod.columns = ['var_code', 'var_name', 'val_code', 'val_name']
grupos = iter(cod.groupby('var_code'))

pd.concat(map(lambda x: remove_nans(x[1]), grupos)).sort_index().reset_index(drop=True)

cod.groupby('var_code').apply(lambda x: x[~x['val_code'].isna()]).index

cod.groupby('var_code').apply(remove_nans).reset_index(0, drop=True).sort_index().reset_index(drop=True)
cod.groupby('var_code').get_group('ICSE')
grupo = cod[cod.var_code == 'ICSE']
grupo.loc[~grupo[['val_code', 'val_name']].isna().any(axis=1), ['var_name', 'val_code', 'val_name']]

def remove_nans(df):
    ## Si la variable es numérica, retornamos el df
    if len(df) == 1:
        return df
    return df[~df[['val_code', 'val_name']].isna().any(axis=1)]


grupo[~grupo[['val_code', 'val_name']].isna().any(axis=1)].join(cod[['var_name']])
cod.groupby('var_code')


ValEtiq(val_df).val_names()
ValEtiq(val_df).val_codes()

ValEtiq(val_df).val_dict()
class VarNoEtiq():

cod = cod.reset_index()
cod.columns = ['var_code', 'var_name', 'val_code', 'val_name']
var = cod.groupby('var_code').apply(VarEtiq)
groups = cod.groupby('var_code')
c = cod.iloc[cod.groupby('var_code').groups['TEL']]

cod.groupby('var_code').apply(ValEtiq)

var['ICSE'].val_codes()
apply(ValEtiq, )        
    
cod.loc['ap36',['Etiqueta.1', 'Códigos']].reset_index(drop=True).set_index('Códigos')['Etiqueta.1'].to_dict()

cod.loc['ap36'].reset_index().Variable.unique()[0]

c = cod.loc['ap36'].reset_index()
c.columns = ['var_code', 'var_name', 'val_code', 'val_name']

c.var_code.unique()[0]

pd
pd.Categorical(c.val_code, ordered=True, categories=c.val_code)


c.columns = ['var_code', 'var_name', 'val_code', 'val_names']
var = VarEtiq(c)
ValEtiq(pd.DataFrame())


var.val_names()
var.val_codes
var.df
VarEtiq()


pd.Categorical(c.val_code, ordered=True, categories=c.val_code)
pd.Categorical()

from pandas.api.types import CategoricalDtype
cat = CategoricalDtype(categories=['a', 'b', 'c'])
cat.add_categories([3])
c.empty

pd.DataFrame(columns=['a', 'b'])

df1 = pd.DataFrame({'grupo':['a','b'], 'values':[1,2]})
df1 = pd.DataFrame({'grupo':['a','b', 'a'], 'values':[1,2,3]})
df1.groupby('grupo').apply(lambda x: x.iloc[x.values==1,])

df1 = pd.DataFrame({'grupo':['a','b', None], 'values':[None,2,3]})