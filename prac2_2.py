# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 19:08:54 2022

@author: julian
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def heatmap(df):
    correlation_matrix = df.corr().round(2)
    print(correlation_matrix)
    sns_h = sns.heatmap(data=correlation_matrix, annot=True)
    sns_h.figure.savefig("Mapa de calor de correlacion")
    sns_h.figure.clear()

def plotXY(df):
    for i in range(0,8):
        plt_disp = plt.scatter(df.iloc[:,i],df.iloc[:,8])
        plt.xlabel(df.columns[i])
        plt.ylabel(df.columns[8])
        cadena = "plt"+str(i)+"_"+str(8)+".png"
        print(cadena)
        plt_disp
        plt_disp.figure.clear()
        
        
if __name__=='__main__':
    df1 = pd.read_csv("data_train.csv", sep=',', engine='python')
    df2 = pd.read_csv("medianHouseValue_train.csv", sep=',', engine='python')
    df = pd.concat([df1,df2], axis=1)
    print("El DataFrame con el 80 por ciento de los datos: \n", df)
    #plt.scatter(df.iloc[:,7],df.iloc[:,8])
    #plt.xlabel(df.columns[7])
    #plt.ylabel(df.columns[8])
    #plt.show()
    heatmap(df)
    plotXY(df)
    