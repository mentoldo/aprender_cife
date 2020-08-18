# -*- coding: utf-8 -*-
from pandas.api.types import CategoricalDtype
import pandas as pd

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
        assert list(val_df.columns) == ['val_code', 'val_name'], "df.columns no es ['val_code', 'val_name']"
        
        self.val_df = val_df[['val_code', 'val_name']]
        self.ordered = ordered
    
    def val_codes(self):
        return CategoricalDtype(categories=self.val_df.val_code, ordered=self.ordered)
    
    def val_names(self):
        return CategoricalDtype(categories=self.val_df.val_name, ordered=self.ordered)
    
    def val_dict(self):
        return self.val_df.set_index('val_code')['val_name'].to_dict()
    
    
class Etiq():
    '''Tiene información sobre las etiquetas'''
    def __init__(self, cod):
        di = self._construct_VarEtiq(cod)
        etiqs = {k: v for d in di for k, v in d.items()} 
        self.var = etiqs
        
    
    def _construct_VarEtiq(self, df):
        grupos = iter(df.groupby('var_code'))
    
        def _get_attrib(x):
            var_code = x[0]
            var_name, df_cod = self._extract_attrib(x[1])
            return {var_code:VarEtiq(var_code, var_name, df_cod)}

        return map(_get_attrib, grupos)
    
    def _extract_attrib(self, grupo):
        '''
        Extrae los atributos para pasar a var etiq
        '''
        val_name = grupo.var_name.iloc[0]
        if len(grupo) > 1:    
            df_cod = grupo[['val_code', 'val_name']].reset_index(drop=True)
        else:
            df_cod = None
        return (val_name,
                df_cod)
#%%
# from src.preprocesing import prepr_code
# cod = prepr_code('./docs/dicc_secundaria.xlsx')
# e = Etiq(cod)
# #%%

# cod = prepr_code('./docs/dicc_secundaria.xlsx')
# e = Etiq(cod)
# type(e)
# e.var['ap3a'].vals.val_dict()
