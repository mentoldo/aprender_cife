# -*- coding: utf-8 -*-
import os
import zipfile

if not os.path.exists('./data/aprender2017-secundaria-12.csv'):
    with zipfile.ZipFile('./data/aprender2017-secundaria-12.zip', 'r') as zip_ref:
        zip_ref.extractall('./data/')

