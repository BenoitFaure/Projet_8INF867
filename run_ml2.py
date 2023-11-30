import datetime as dt
import yfinance as yf

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler


import pandas
import talib
import pandas_ta as ta

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

import warnings

# Ignorer les avertissements liés à is_sparse dans sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")




def get_company_data(name):
    start = dt.datetime(2005,1,1)
    end = dt.datetime(2022,1,1)
    data = yf.download(name, start = start, end=end)
    return data

def apply_new_features(data):
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)


    data.ta.bbands(length=20, append=True)
    data.ta.macd(append=True)


    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Rate_of_Change'] = data['Close'].pct_change(periods=10)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    
   
    return data

def fill_data_mean(data):
    return data.fillna(data.mean())

def normalize(data,target="Close"):
    y = data[[target]]
    X = data.drop(target, axis=1)
    scaler = StandardScaler()
    if not X.empty:
        X[X.columns] = scaler.fit_transform(X)
    return X,y

def split_datas(X,y,nb_predict=500):
    
    X_train = X[:-nb_predict]
    y_train = y[:-nb_predict]
    X_val = X[-nb_predict:]
    y_val = y[-nb_predict:]
    return X_train,y_train,X_val,y_val


def train_and_eval_model(X_train,y_train,X_val,y_val,model,comp_name):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_val, y_val)
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")
    
    train_pred = model.predict(X_train)
    plt.subplot(1,2,1)
    plt.plot(range(y_train.shape[0]),train_pred.ravel(), color='blue', label='Predicted')
    plt.plot(range(y_train.shape[0]),y_train.values.ravel(), color='red', label='Actual')
    plt.title('Train Data '+ comp_name)
    plt.legend()
    
    val_pred = model.predict(X_val)
    plt.subplot(1,2,2)
    plt.plot(range(y_val.shape[0]),val_pred.ravel(), color='blue', label='Predicted')
    plt.plot(range(y_val.shape[0]),y_val.values.ravel(), color='red', label='Actual')
    plt.title('Val Data '+ comp_name)
    plt.legend()
    
    
    plt.show()
    
    
    



if __name__ == "__main__":
    
    compagnies = ['TSLA','BAC','JNJ','MSFT','XOM','GE','BIIB']
    for comp in compagnies:
        data = get_company_data(comp)
        data = apply_new_features(data)
        data = fill_data_mean(data)
        X,y = normalize(data)
        X_train,y_train,X_val,y_val = split_datas(X,y,nb_predict=500)
        model = SVR(degree=3,kernel='rbf', C=1, gamma='auto')
        train_and_eval_model(X_train,y_train,X_val,y_val,model,comp)
        
        
    
    

