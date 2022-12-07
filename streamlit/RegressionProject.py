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

database = pd.read_csv('fifa_23_280922.csv') 

database = database.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna(axis=1, how='all')

database  = database.drop('Taille', axis = 1)

database = database.drop('Id', axis = 1)

X = database.drop('Potentiel', axis = 1)

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