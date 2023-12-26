# import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.discriminant_analysis import StandardScaler


def janelaDeslizante(vetor, janelaEntrada, janelaSaida):
    """ 
    Calcula a matriz de conhecimento

    Args:
        vetor: Array com os valores da série temporal.
        janelaEntrada: int com o tamanho da janela de entrada.
        JanelaSaida: int com o tamanho da janela de saída.

    Returns:
        X: matriz com as entradas do modelo tem tamanho (numero de amostras X tamanho da janela entrada)
        y: matriz com as saídas do modelo tem tamanho (numero de amostras X tamanho da janela de saida)
    """
    #### YEO johnson
    transformacao, lambda_param = stats.yeojohnson(vetor)
    X_y = transformacao
    
    ## Padronização
    scaler = StandardScaler()
    f = np.array(X_y).reshape(-1,1)
    X_p = scaler.fit_transform(f)
    
    X_p = np.array(X_p).flatten()
    
    num_amostras = len(X_p) - janelaEntrada - janelaSaida + 1
    X = np.zeros((num_amostras, janelaEntrada))
    y = np.zeros((num_amostras, janelaSaida))
    
    for i in range(num_amostras):
        X[i] = X_p[i : i + janelaEntrada]
        y[i] = X_p[i + janelaEntrada : i + janelaEntrada + janelaSaida]

    return X, y
  
  
  
def dividirConjuntoDados(X,y, tamanhoTeste=0.2):
  """
    Divide o conjunto de dados em treino e teste dado a porcentagem para o teste, sem realizar o embaralhamento
    
    Args:
        X: matriz com as entradas do modelo tem tamanho (numero de amostras X tamanho da janela entrada)
        Y: matriz com as saídas do modelo tem tamanho (numero de amostras X tamanho da janela de saida)
        tamanhoTeste: float porcentagem referente ao tamanho do teste

    Returns:
        y_pred: matriz com as previsoes realizadas pelo modelo ajustado
    """
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tamanhoTeste, random_state=42, shuffle=False)
  
  return X_train,y_train,X_test,y_test



def ajustarModelo (X_train, y_train, X_test):
  """
    Ajusta o modelo de LSTM
    
    Args:
        X_train: matriz com as entradas do modelo tem tamanho (numero de amostras X tamanho da janela entrada)
        y_train: matriz com as saídas do modelo tem tamanho (numero de amostras X tamanho da janela de saida)
        X_test: matriz com as entradas para teste do modelo tem tamanho (numero de amostras X tamanho da janela entrada)

    Returns:
        y_pred: matriz com as previsoes realizadas pelo modelo ajustado
    """
    
  model = Sequential() 
  model.add(LSTM(units=64, input_shape=(X.shape[1], 1),return_sequences=True))
  
  for _ in range(1): 
      model.add(LSTM(units=64, return_sequences=True)) 
      model.add(Dropout(0.2))
   
  model.add(LSTM(units=64))
  model.add(Dropout(0.2))  # Regularização Dropout
  model.add(Dense(units=1))  # Saída única para previsão de valor futuro

  # Compilar o modelo
  model.compile(optimizer='adam', loss='mean_squared_error') 

  # Treinar o modelo
  model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=2) # type: ignore  

  # # Fazer previsões
  # predictions = model.predict(X_test)
 
  # # Fazer previsões 
  # y_pred = model.predict(X_test)
 
  
  return model



## MODELO INDIVIDUAL 
dados = pd.read_excel('../codigos/data/2019.xlsx')
ativos = dados.columns[1:]
ativos = ['WEGE3','YDUQ3']

for ativo in ativos:
  X,y = janelaDeslizante(dados[ativo], 22,1)

  X_train,y_train, X_test, y_test = dividirConjuntoDados(X,y)

  model = ajustarModelo(X_train, y_train, X_test)

  model.save(f'models/individuais/{ativo}.h5')