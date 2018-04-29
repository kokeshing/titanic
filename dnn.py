import os
import sys
import random
from datetime import datetime, timedelta
import numpy as np
from keras import optimizers, callbacks, initializers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, BatchNormalization

import pandas as pd


# 特徴の次元数
IN_NEUR  = 4
# バッチサイズ
BATCH_NM = 32
# エポック数
EPOCH_NM = 1000
log_filepath = './log/'
result_dir = './result/'

def min_max_normalization(array, axis=None):
    min = array.min(axis=axis, keepdims=True)
    max = array.max(axis=axis, keepdims=True)
    result = (array-min)/(max-min)
    return result

def create_model():
    #prelu = advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

    # Model
    model = Sequential()
    #input layer
    model.add(Dense(6, input_shape=(6,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # hidden layers
    model.add(Dense(6))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(6))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(3))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    # output layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.5),
                  metrics=['accuracy'])
    return model

def mktraindata():
    df = pd.read_csv("usetrain.csv")
    survived_train = []
    unsurvived_train = []
    suv_num = 0
    unsuv_num = 0

    pclass_arr = np.array([df.at[index, "Pclass"] for index in range(0, 714)])
    sex_arr    = np.array([df.at[index, "Sex"] for index in range(0, 714)])
    age_arr    = np.array([df.at[index, "Age"] / 100.0 for index in range(0, 714)])
    sibsp_arr  = np.array([df.at[index, "SibSp"] / 10.0 for index in range(0, 714)])
    parch_arr  = np.array([df.at[index, "Parch"] / 10.0 for index in range(0, 714)])
    fare_arr   = np.array([np.log(df.at[index, "Fare"]) if df.at[index, "Fare"] > 0 else 0 for index in range(0, 714)])
    class_arr  = np.array([df.at[index, "Survived"] for index in range(0, 714)])

    row0 = min_max_normalization(pclass_arr)
    row1 = sex_arr
    row2 = age_arr
    row3 = sibsp_arr
    row4 = parch_arr
    row5 = min_max_normalization(fare_arr)

    for i in range(0,714):
        data = np.hstack((row0[i], row1[i], row2[i], row3[i], row4[i], row5[i]))
        # survived
        if (class_arr[i] == 1):
            survived_train.append(data)
        # unsurvived
        else:
            unsurvived_train.append(data)

    fewest = min([len(survived_train), len(unsurvived_train)])
    x_train = random.sample(survived_train, fewest) + random.sample(unsurvived_train, fewest)
    y_train = [1 for i in range(0, fewest)] + [0 for i in range(0, fewest)]

    if len(x_train) == len(y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]

        return x_train, y_train
    else:
        printf("ラベルとデータの長さが違います")
        sys.exit(1)

if __name__=='__main__':
    model = create_model()

    cp_cb = callbacks.ModelCheckpoint(
        filepath = './result/model{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-vloss{val_loss:.3f}-vacc{val_acc:.3f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto')

    tb_cb = callbacks.TensorBoard(
        log_dir=log_filepath,
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    es_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=0,
        mode='auto')

    x_train , y_train = mktraindata()

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_NM,
        epochs=EPOCH_NM,
        verbose=1,
        callbacks=[cp_cb, tb_cb, es_cb],
        validation_split=0.2,
        shuffle=False)

    model.save(os.path.join(result_dir, 'trained_model.h5'))