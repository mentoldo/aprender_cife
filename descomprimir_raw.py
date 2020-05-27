# -*- coding: utf-8 -*-
import os
import zipfile

if not os.path.exists('./data/aprender2017-secundaria-12.csv'):
    with zipfile.ZipFile('./data/aprender2017-secundaria-12.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')



#%%
p = os.path.abspath('/home/psyche/Escritorio/aprender_cife/file.pdf')

os.path.join('psyche', 'Escritorio')

os.pardir

grandparent_dir = os.path.abspath(  # Convert into absolute path string
    os.path.join(  # Current file's grandparent directory
        os.path.join(  # Current file's parent directory
            os.path.dirname(  # Current file's directory
                os.path.abspath(__file__)  # Current file path
            ),
            os.pardir
        ),
        os.pardir
    )
)

print grandparent_dir