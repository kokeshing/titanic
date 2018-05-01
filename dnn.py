import os
import sys
import random
from datetime import datetime, timedelta
import numpy as np
from keras import optimizers, callbacks, initializers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, BatchNormalization

import pandas as pd

# バッチサイズ
BATCH_NM = 20
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
    # Model
    model = Sequential()
    #input layer
    model.add(Dense(7, input_shape=(7,)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # hidden layers
    model.add(Dense(7))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(7))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(4))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))

    model.add(Dense(2))
    model.add(BatchNormalization())
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.80, momentum=0.1),
                  metrics=['accuracy'])
    return model

def mktraindata():
    df = pd.read_csv("usetrain2.csv")
    survived_train = []
    unsurvived_train = []
    suv_num = 0
    unsuv_num = 0

    pclass_arr = np.array([df.at[index, "Pclass"] for index in range(0, 712)])
    sex_arr    = np.array([df.at[index, "Sex"] for index in range(0, 712)])
    age_arr    = np.array([df.at[index, "Age"] / 100.0 for index in range(0, 712)])
    sibsp_arr  = np.array([df.at[index, "SibSp"] / 10.0 for index in range(0, 712)])
    parch_arr  = np.array([df.at[index, "Parch"] / 10.0 for index in range(0, 712)])
    fare_arr   = np.array([np.log(df.at[index, "Fare"]) if df.at[index, "Fare"] > 0 else 0 for index in range(0, 712)])
    embarked   = np.array([df.at[index, "Embarked"] for index in range(0, 712)])
    class_arr  = np.array([df.at[index, "Survived"] for index in range(0, 712)])

    row0 = min_max_normalization(pclass_arr)
    row1 = sex_arr
    row2 = age_arr
    row3 = sibsp_arr
    row4 = parch_arr
    row5 = min_max_normalization(fare_arr)
    row6 = embarked

    for i in range(0,712):
        data = np.hstack((row0[i], row1[i], row2[i], row3[i], row4[i], row5[i], row6[i]))
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
        validation_split=0.25,
        shuffle=False)

    model.save(os.path.join(result_dir, 'trained_model.h5'))