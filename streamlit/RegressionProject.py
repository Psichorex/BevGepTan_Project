# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import streamlit as st

import numpy as np

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split, cross_validate

import pandas as pd

import matplotlib.pyplot as plt

database = pd.read_csv('fifa_23_280922.csv') 

database = database.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1, how='all')

database  = database.drop('Taille', axis = 1)

database = database.drop('Id', axis = 1)

database = database.drop('Mauvais pied', axis = 1)

database = database.drop('Gestes techniques', axis = 1)

X = database.drop('Potentiel', axis = 1)

features = X.keys()

X.head()

y = database['Potentiel']

y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)

lasso = Lasso(max_iter=10000, tol= 0.001, alpha = 1.0)

lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

st.markdown('Prediction based on X_test: ')

st.dataframe(X_test)

st.markdown('Predictions player potential: ')

st.dataframe(y_pred)

crossVal = cross_validate(lasso, X, y, cv=10, return_train_score=True)

st.markdown('Train Score: ')  
st.markdown(np.average(crossVal['train_score']))
st.markdown('Test Score: ')
st.markdown(np.average(crossVal['test_score']))

st.markdown('Try yourself')

stat1 = st.slider('Stat', min_value=0, max_value=99, key=0)
stat2 = st.slider('Stat', min_value=0, max_value=99, key=1)
stat3 = st.slider('Stat', min_value=0, max_value=99, key=2)
stat4 = st.slider('Stat', min_value=0, max_value=99, key=3)
stat5 = st.slider('Stat', min_value=0, max_value=99, key=4)
stat6 = st.slider('Stat', min_value=0, max_value=99, key=5)
stat7 = st.slider('Stat', min_value=0, max_value=99, key=6)
stat8 = st.slider('Stat', min_value=0, max_value=99, key=7)
stat9 = st.slider('Stat', min_value=0, max_value=99, key=8)
stat10 = st.slider('Stat', min_value=0, max_value=99, key=10)
stat11 = st.slider('Stat', min_value=0, max_value=99, key=11)
stat12 = st.slider('Stat', min_value=0, max_value=99, key=12)
stat13 = st.slider('Stat', min_value=0, max_value=99, key=13)
stat14 = st.slider('Stat', min_value=0, max_value=99, key=14)
stat15 = st.slider('Stat', min_value=0, max_value=99, key=15)
stat16 = st.slider('Stat', min_value=0, max_value=99, key=16)
stat17 = st.slider('Stat', min_value=0, max_value=99, key=17)
stat18 = st.slider('Stat', min_value=0, max_value=99, key=18)
stat19 = st.slider('Stat', min_value=0, max_value=99, key=19)
stat20 = st.slider('Stat', min_value=0, max_value=99, key=20)
stat21 = st.slider('Stat', min_value=0, max_value=99, key=21)
stat22 = st.slider('Stat', min_value=0, max_value=99, key=22)
stat23 = st.slider('Stat', min_value=0, max_value=99, key=23)
stat24 = st.slider('Stat', min_value=0, max_value=99, key=24)
stat25 = st.slider('Stat', min_value=0, max_value=99, key=25)
stat26 = st.slider('Stat', min_value=0, max_value=99, key=26)
stat27 = st.slider('Stat', min_value=0, max_value=99, key=27)
stat28 = st.slider('Stat', min_value=0, max_value=99, key=28)
stat29 = st.slider('Stat', min_value=0, max_value=99, key=29)
stat30 = st.slider('Stat', min_value=0, max_value=99, key=30)
stat31 = st.slider('Stat', min_value=0, max_value=99, key=31)
stat32 = st.slider('Stat', min_value=0, max_value=99, key=32)
stat33 = st.slider('Stat', min_value=0, max_value=99, key=33)
stat34 = st.slider('Stat', min_value=0, max_value=99, key=34)
stat35 = st.slider('Stat', min_value=0, max_value=99, key=35)
stat36 = st.slider('Stat', min_value=0, max_value=99, key=36)

player = [stat1,stat2,stat3,stat4,stat5,stat6,stat7,stat8,stat9,stat10,
stat11,stat12,stat13,stat14,stat15,stat16,stat17,stat18,stat19,stat20,
stat21,stat22,stat23,stat24,stat25,stat26,stat27,stat28,stat29,stat30,
stat31,stat32,stat33,stat34,stat35,stat36]

array = np.array(player)
df = pd.DataFrame(array.reshape(-1, len(array)), columns=features)
playerPredict = lasso.predict(df)
st.markdown(playerPredict)