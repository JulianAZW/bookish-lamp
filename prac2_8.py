# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 05:12:16 2022

@author: julia
"""

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import SGDRegressor
import statistics as st
from  sklearn import preprocessing

mse_list1 = list()
r2_list1 = list()

mse_list2 = list()
r2_list2 = list()

mse_list3 = list()
r2_list3 = list()


mse_list1S = list()
r2_list1S = list()

mse_list2S = list()
r2_list2S = list()

mse_list3S = list()
r2_list3S = list()

mse_list1R = list()
r2_list1R = list()

mse_list2R = list()
r2_list2R = list()

mse_list3R = list()
r2_list3R = list()



# Mejor regresion con el 80% de los datos



def regre3Standard(x,y):

    y = np.array(y)
    # Se crea el objeto linear regression
    regr = SGDRegressor(learning_rate = 'constant', eta0 = 1*(10)**-6, max_iter= 100000)
    
    #Conversión de las variables de la ecuación original a polinomio de grado 3
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)

    
    #~ ####Escalado de los datos####
    #~ #Standard Scaler
    x_poly_standard_scaler = preprocessing.StandardScaler().fit_transform(x_poly)


    regr.fit(x_poly_standard_scaler, y.ravel())
    y_poly_pred = regr.predict(x_poly_standard_scaler)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    print ('Regresión polinomial estocástico grado 3 con escalado de datos Standard\nmse: {} r2: {}'.format(mse, r2))
    mse_list3S.append(mse)
    r2_list3S.append(r2)






if __name__=='__main__':
    
  

    print("*************************************************************")
    df1 = pd.read_csv("data_train.csv", sep=',', engine='python')
    df2 = pd.read_csv("medianHouseValue_train.csv", sep=',', engine='python')
    x = df1
    y = df2
    regre3Standard(x.values, y.values)

    print("*************************************************************")
    

  
    
    print("Lista del mse de regresion polinomial 3 con escalado de datos standard.: ", mse_list3S)
    print("Lista del r2 de regresion polinomial 3 con escalado de datos standard: ", r2_list3S)
    print("Promedio de los mse de regresion polinomial 3 con escalado de datos standard: ", st.mean(mse_list3S))
    print("Promedio de los r2 de regresion polinomial 3 con escalado de datos standard: ", st.mean(r2_list3S))


    print("*************************************************************")
    
    