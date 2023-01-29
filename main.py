import warnings
import datetime
warnings.filterwarnings('ignore')
import pickle
from iqoptionapi.stable_api import IQ_Option
iq = IQ_Option(f"marcosmordefronha123@gmail.com", "Meuovo123!")
iq.connect()  # connect to iqoption
iq.get_all_init()
import os
# definindo a variavel par
par = 'AUDCAD-OTC'
bet_money = 5
# definindo o time frame de 1m
time_frame = 60
nrtentativas = 0
# definindo um sleep de 60s
sleep_time = 62
trade = True
treinar = True
# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
from keras.callbacks import EarlyStopping
import keras.layers.normalization as normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import time
ganhos_compra = []
ganhos_venda = []
trade = False
tf.function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    reduce_retracing=True,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_relax_shapes=None,
    experimental_compile=None,
    experimental_follow_type_hints=None
)
#tf.executing_eagerly()

tf.config.run_functions_eagerly(True)
@tf.function
def fn():
  with tf.init_scope():
    print(tf.executing_eagerly())
  print(tf.executing_eagerly())
fn()
galo = 0

def prediction(par,iq,time_frame,time):
    velas = iq.get_candles(par, time_frame, 60, time.time())
    data = pd.DataFrame(velas)
    X = data[["open", "close", "min", "max", "volume"]]
    # Adicionando indicadores técnicos
    data['SMA9'] = data['close'].rolling(window=2).mean()
    data['SMA21'] = data['close'].rolling(window=9).mean()
    data['BB_upper'] = data['close'].rolling(window=10).mean() + 1 * data['close'].rolling(window=10).std()
    data['BB_lower'] = data['close'].rolling(window=10).mean() - 1 * data['close'].rolling(window=10).std()
    X = data[["open", "close", "min", "max", "volume", "SMA9", "SMA21", "BB_upper", "BB_lower"]]
    # Preparando os dados para treinamento
    # Normalizando os dados
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # Criando um dataframe com as colunas de dados normalizados
    df = pd.DataFrame(X, columns=["open", "close", "min", "max", "volume", "SMA9", "SMA21", "BB_upper", "BB_lower"])
    # Criando a coluna de labels com os valores de compra (1) e venda (0)
    df['label'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
    df.isna().sum()
    df = df.fillna(0)
    # Dividindo os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2)
    model = load_model('model222.h5')
    previsao = model.predict(X_test)
    score = model.evaluate(X_test, y_test, verbose=2)
    print('Test accuracy:', score[1])
    print(previsao[0][0])
    print(previsao[0][1])
    return previsao[0][0],previsao[0][1]


# Definindo a lógica para realizar as operações
while True:
   #while datetime.datetime.now().second > 1:
   #    xd = "xd"
    if treinar:
        treinar = False
        velas = iq.get_candles(par, time_frame, 1000, time.time())
        data = pd.DataFrame(velas)
        X = data[["open", "close", "min", "max", "volume"]]
        # Adicionando indicadores técnicos
        data['SMA9'] = data['close'].rolling(window=2).mean()
        data['SMA21'] = data['close'].rolling(window=9).mean()
        data['BB_upper'] = data['close'].rolling(window=10).mean() + 1 * data['close'].rolling(window=10).std()
        data['BB_lower'] = data['close'].rolling(window=10).mean() - 1 * data['close'].rolling(window=10).std()
        X = data[["open", "close", "min", "max", "volume", "SMA9", "SMA21", "BB_upper", "BB_lower"]]
        # Preparando os dados para treinamento
        # Normalizando os dados
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        # Criando um dataframe com as colunas de dados normalizados
        df = pd.DataFrame(X, columns=["open", "close", "min", "max", "volume", "SMA9", "SMA21", "BB_upper", "BB_lower"])
        # Criando a coluna de labels com os valores de compra (1) e venda (0)
        df['label'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)
        df.isna().sum()
        df = df.fillna(0)
        # Dividindo os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2)
        # Criando o modelo de Random Forest
        # Criando o modelo
        learning_rate = 0.07
        epochs = 500
        batch_size = 128
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(
           0.01)))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='softmax'))
       # model.add(Dropout(0.2))
        model.compile(Adam(learning_rate=learning_rate, decay=5e-5), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        # Treinar o modelo
        history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[
                EarlyStopping(monitor='accuracy', patience=40, verbose=2, mode='max'),
                ModelCheckpoint('model222.h5', monitor='accuracy', save_best_only=True, mode='max', reduce_retracing=True)
            ]
        )
    while datetime.datetime.now().second != 1:
        xd = "xd"
    previsao = prediction(par,iq,time_frame,time)
    # Definindo que o treinamento não é mais necessário
    if previsao[0] > 0.51:
       blablabla = iq.get_balance()
       while datetime.datetime.now().second > 5:
           xd = "xd"
       check, id = iq.buy(bet_money, par, "call", 1)
       print("call")
       trade = True
       treinar = False
       time.sleep(62)
    elif previsao[0] <= 0.49:
       blablabla = iq.get_balance()
       while datetime.datetime.now().second > 5:
           xd = "xd"
       check, id = iq.buy(bet_money, par, "put", 1)
       # Realizando a operação de venda
       trade = True
       treinar = False
       time.sleep(62)
       print("put")
    else:
       trade = False
       time.sleep(15)
    if trade:
        betsies = iq.get_balance()
        vaisefude = betsies - blablabla
        if vaisefude > 1:
            print("WIN")
            if previsao[0] > 0.52:
                treinar = False
                ganhos_compra.append(vaisefude)
                model.save(os.path.join('model222.h5'))
            elif previsao[0] <= 0.49:
                treinar = False
                ganhos_venda.append(vaisefude)
                model.save(os.path.join('model222.h5'))
            galo = 0
            # adicionando dados nos resultados de compra
            bet_money = 5
            time.sleep(120)
        else:
            galo += 1
            print('GALE =>', galo)
            if galo == 1:
                bet_money = 5 * 2.15
            elif galo == 2:
                bet_money = 5 * 4.15
            elif galo > 2:
                galo = 0
                print("loss")
                treinar = True
                galo = 0
                bet_money = 5


  #     # Reiniciando o número de tentativas
  #     nrtentativas = 0

  # # Dormindo por 60 segundos
  # time.sleep(sleep_time)
