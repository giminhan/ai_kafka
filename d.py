import pandas as pd
import tensorflow as tf
import numpy as np
import os
import datetime

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D
from tensorflow.keras.layers import LSTM, MaxPooling1D, Reshape, Input, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import keras_tuner as kt


# 세부 파라미터 조정
num_classes = 10
batch_size = 128  # 64~128
epochs = 100  # 10~50
learning_rate = 0.002
tf.random.set_seed(0)
np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))  # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

# 라벨링 mnist 는 0~9 총 10개
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

input_shape = Input(shape=[X_train.shape[1], X_train.shape[2], X_train.shape[3]])
activate_relu = tf.keras.layers.LeakyReLU()


# 범용 활성화함수
# activation 입니다 False 라고 한 이유는 각 activation function parameter False 는 사용안함 True 하면 사용
def activation_optional(input_size, leaky_relu=True):
    # d = Conv1D(10, 25, padding="same")(input_size)
    norm = BatchNormalization()(input_size)
    if leaky_relu:
        activate = tf.keras.layers.LeakyReLU(alpha=0.3)(norm)
    else:
        activate = tf.keras.layers.ReLU()(norm)
    return MaxPooling2D(3)(activate)


# cnn1d : conv1d -> conv1d -> (bn+maxpooling1d) -> flatten -> fullyconnection -> softmax
# 만약 학습 안되면 dropout 주석 해제하고
# finally_dense(dense1) -> finally_dense(dropout_concat) 하고 다시 진행
def lstm_cnn_modeling(hp):
    cnn1d = Conv2D(filters=hp.Int('conv_1_filter', min_value=10, max_value=20, step=16),
                   kernel_size=hp.Choice('conv_1_kernel', values=[20, 50]), padding="same", activation=activate_relu)(input_shape)
    cnn1d = Conv2D(filters=hp.Int('conv_2_filter', min_value=10, max_value=20, step=16),
                   kernel_size=hp.Choice('conv_2_kernel', values=[20, 50]), padding="same", activation=activate_relu)(cnn1d)

    bm = activation_optional(cnn1d)
    # lstm = (LSTM(10, activation="tanh", return_sequences=True))(cnn1d)
    flatten = Flatten()(bm)
    dense1 = Dense(units=hp.Int('units', min_value=32, max_value=64, step=16), activation=activate_relu)(flatten)
    #dropout_concat = (Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05, )))(
    #    dense1)
    finally_dense = (Dense(num_classes, activation='softmax'))(dense1)
    k_model = Model(input_shape, finally_dense)

    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    adam = Adam(amsgrad=True)
    k_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    return k_model


def main():
    # RandomSearch
    from kerastuner.tuners import RandomSearch
    tuner = RandomSearch(lstm_cnn_modeling,  # HyperModel
                         objective='val_acc',  # 최적화할 하이퍼모델
                         max_trials=20,
                         executions_per_trial=1,  # 각 모델별 학습 회수
                         directory='K',  # 사용된 parameter 저장할 폴더
                         project_name='TEST')  # 사용된 parameter 저장할 폴더

    # 튜너 학습
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=1, validation_data=(X_test, y_test), callbacks=[callback], batch_size=340)

    # 최적의 학습 모델 출력
    models = tuner.get_best_models(num_models=1)[0]
    print(models)
    
    # 일단 주석처리
    print(models.save('cnn_64_batch_dropout.h5'))

    # 학습 결과 출력
    tuner.results_summary()

    # confusion matrix making text file
    # def print_matrix():
    #    confusion = print_confusion_matrix_v2(prediction_result, y_test)
    #    logw(file, f'Confusion Matrix -> \n' + np.array2string(confusion))

    # print_matrix()


main()
