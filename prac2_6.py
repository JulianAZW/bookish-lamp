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


mse_list1 = list()
r2_list1 = list()

mse_list2 = list()
r2_list2 = list()

mse_list3 = list()
r2_list3 = list()

#Regresiones de TODAS las variables


def regre1(x,y):
    # 
    y = np.array(y)
    # Se crea el objeto linear regression
    regr = SGDRegressor(learning_rate = 'constant', eta0 = 1*(10)**-6, max_iter= 1000000)
    # Se entrena el modelo
    regr.fit(x, y.ravel())
    y_pred = regr.predict(x)
    
    #Lo ploteamos
    #plt.plot(x,y_pred, color='r')
    #plt.scatter(x, y, s=10)
    #plt.show()
    
    #Cálculo del error cuadrado medio y r2
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("Regresion lineal\n mse: {} r2: {}".format(mse, r2))
    mse_list1.append(mse)
    r2_list1.append(r2)

def regre2(x,y):
    # 
    y = np.array(y)
    # Se crea el objeto linear regression
    regr = SGDRegressor(learning_rate = 'constant', eta0 = 1*(10)**-6, max_iter= 100000)
    
    #Conversión de las variables de la ecuación original a polinomio de grado 2
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)

    regr.fit(x_poly, y.ravel())
    y_poly_pred = regr.predict(x_poly)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    print ('Regresión polinomial estocástico grado 2\nmse: {} r2: {}'.format(mse, r2))
    mse_list2.append(mse)
    r2_list2.append(r2)

def regre3(x,y):
    # 
    y = np.array(y)
    # Se crea el objeto linear regression
    regr = SGDRegressor(learning_rate = 'constant', eta0 = 1*(10)**-6, max_iter= 100000)
    
    #Conversión de las variables de la ecuación original a polinomio de grado 3
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)

    regr.fit(x_poly, y.ravel())
    y_poly_pred = regr.predict(x_poly)
    mse = mean_squared_error(y, y_poly_pred)
    r2 = r2_score(y, y_poly_pred)
    print ('Regresión polinomial estocástico grado 3\nmse: {} r2: {}'.format(mse, r2))
    mse_list3.append(mse)
    r2_list3.append(r2)

if __name__=='__main__':            

    for i in range (1,11): 
        print("Se hizo el numero " + str(i) + " de regresion lineal.")
        df1 = pd.read_csv("data_validation_train"+str(i)+".csv", sep=',', engine='python')
        df2 = pd.read_csv("medianHouseValue_validation_train"+str(i)+".csv", sep=',', engine='python')
        x = df1
        y = df2
        regre1(x.values, y.values)
        
    for i in range (1,11):
        print("Se hizo el numero " + str(i) + " de regresion polinomial 2.")
        df1 = pd.read_csv("data_validation_train"+str(i)+".csv", sep=',', engine='python')
        df2 = pd.read_csv("medianHouseValue_validation_train"+str(i)+".csv", sep=',', engine='python')
        x = df1
        y = df2
        regre2(x.values, y.values)

    for i in range (1,11):
        print("Se hizo el numero " + str(i) + " de regresion polinomial 3.")
        df1 = pd.read_csv("data_validation_train"+str(i)+".csv", sep=',', engine='python')
        df2 = pd.read_csv("medianHouseValue_validation_train"+str(i)+".csv", sep=',', engine='python')
        x = df1
        y = df2
        regre3(x.values, y.values)

    print()

    print("REGRESIONES CON TODAS LAS VARIABLES")

    print()

    print("*************************************************************")
    

    print("Lista del mse de regresion lineal: ", mse_list1)
    print("Lista del r2 de regresion lineal: ", r2_list1)
    print("Promedio de los mse de regresion lineal: ", st.mean(mse_list1))
    print("Promedio de los r2 de regresion lineal: ", st.mean(r2_list1))
    
    print("*************************************************************")
    
    print("Lista del mse de regresion polinomial 2: ", mse_list2)
    print("Lista del r2 de regresion polinomial 2: ", r2_list2)
    print("Promedio de los mse de regresion polinomial 2: ", st.mean(mse_list2))
    print("Promedio de los r2 de regresion polinomial 2: ", st.mean(r2_list2))
    
    print("*************************************************************")
    
    print("Lista del mse de regresion polinomial 3: ", mse_list3)
    print("Lista del r2 de regresion polinomial 3: ", r2_list3)
    print("Promedio de los mse de regresion polinomial 3: ", st.mean(mse_list3))
    print("Promedio de los r2 de regresion polinomial 3: ", st.mean(r2_list3))


    print("*************************************************************")