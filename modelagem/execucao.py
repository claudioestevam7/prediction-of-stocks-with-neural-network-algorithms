import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import numpy as np
import pandas as pd

import os

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

  
  
def ajustarModelo (X_train, y_train):
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dense, Dropout
  """
    Ajusta o modelo de LSTM
    
    Args:
        X_train: matriz com as entradas do modelo tem tamanho (numero de amostras X tamanho da janela entrada)
        y_train: matriz com as saídas do modelo tem tamanho (numero de amostras X tamanho da janela de saida)
        X_test: matriz com as entradas para teste do modelo tem tamanho (numero de amostras X tamanho da janela entrada)

    Returns:
        y_pred: matriz com as previsoes realizadas pelo modelo ajustado
        Retorna o modelo para ser salvo e posteriormente realizar as metricas de avaliação
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
  model.fit(X_train, y_train, epochs=500, batch_size=1, verbose=2)

  # # Fazer previsões
  # predictions = model.predict(X_test)

  # # Fazer previsões
  # y_pred = model.predict(X_test)

  
  return model



dados_2019 = pd.read_excel('../codigos/data/2019.xlsx')



janeiro_2019 = dados_2019[:21]
fevereiro_2019 = dados_2019[21:41]
marco_2019 = dados_2019[41:60]
abril_2019 = dados_2019[60:81]
maio_2019 = dados_2019[81:103]
junho_2019 = dados_2019[103:122]
julho_2019 = dados_2019[122:144]
agosto_2019 = dados_2019[144:166]
setembro_2019 = dados_2019[166:187]
outubro_2019 = dados_2019[187:210]
novembro_2019 = dados_2019[210:229]
dezembro_2019 = dados_2019[229:248]

incrementos = [[], janeiro_2019, fevereiro_2019, marco_2019, abril_2019, maio_2019, junho_2019, julho_2019, agosto_2019, setembro_2019, outubro_2019, novembro_2019, dezembro_2019]
meses = ['1. Janeiro', '2. Fevereiro', '3. Março', '4. Abril', '5. Maio', '6. Junho', '7. Julho', '8. Agosto', '9. Setembro', '10. Outubro', '11. Novembro', '12. Dezembro']


janelas = [ '1. jan 2018 - dez 2018   -> jan 2019',
            '2. jan 2018 - jan 2019   -> fev 2019',
            '3. jan 2018 - fev 2019   -> mar 2019',
            '4. jan 2018 - mar 2019   -> abr 2019',
            '5. jan 2018 - abr 2019   -> mai 2019',
            '6. jan 2018 - mai 2019   -> jun 2019',
            '7. jan 2018 - jun 2019   -> jul 2019',
            '8. jan 2018 - jul 2019   -> ago 2019',
            '9. jan 2018 - ago 2019   -> set 2019',
            '10. jan 2018 - set 2019   -> out 2019',
            '11. jan 2018 - out 2019   -> nov 2019',
            '12. jan 2018 - nov 2019   -> dez 2019']


## MODELO INDIVIDUAL com periodo deslizante


for i, janela in enumerate(janelas):
  print(i)
  
  if i == 0:
    dados = pd.read_excel('../codigos/data/2018.xlsx')
    ativos = dados.columns[1:]
  else:
    dados = np.concatenate((dados,incrementos[i]), axis=0)
  dados = pd.DataFrame(dados, columns = ativos.insert(0, 'date'))
  
  
  
  for ativo in ativos:
    if os.path.exists(f'models/individuais/2018/deslizante/{janela}/{ativo}.h5'):
      print("já existe")
      print(f'models/individuais/2018/deslizante/{janela}/{ativo}.h5')
    else:
        
      print(dados[ativo])
      # break
      X,y = janelaDeslizante(np.array(dados[ativo], float), 22,1)

      X_train,y_train = X, y

      model = ajustarModelo(X_train, y_train)
      
      print(f'models/individuais/2018/deslizante/{janela}/{ativo}.h5')
      model.save(f'models/individuais/2018/deslizante/{janela}/{ativo}.h5')