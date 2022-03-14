import tensorflow as tf
import numpy as np
import os
import datetime, random
import keras_tuner as kt
import matplotlib.pyplot as plt 

from confusion_shared import logw, print_confusion_matrix_v2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D
from tensorflow.keras.layers import LSTM, MaxPooling1D, Reshape, Input, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


# 세부 파라미터 조정
num_classes = 10
batch_size = 128  # 64~128
epochs = 10  # 10~50
learning_rate = 0.001
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

fnpath = 'result/'
try:
    os.mkdir(fnpath)
except OSError as e:
    print(f'An error has occurred. Continuing anyway: {e}')

# file create location
filename = f'{os.getcwd()}/result/classification_output.txt'
file = open(filename, 'a+')


def train_file_info():
    logw(file, 'start -> {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
    logw(file, f'Shape checking X_test image: {X_test.shape}')
    logw(file, f'Shape checking X_train image: {X_train.shape}')


# 범용 활성화함수
def activation_optional(input_size, leaky_relu=True):
    # d = Conv1D(10, 25, padding="same")(input_size)
    norm = BatchNormalization()(input_size)
    if leaky_relu:
        activate = tf.keras.layers.LeakyReLU(alpha=0.3)(norm)
    else:
        activate = tf.keras.layers.ReLU()(norm)
    return MaxPooling2D(3)(activate)


# cnn2d : conv2d -> conv2d -> (bn+maxpooling1d) -> flatten -> fullyconnection -> softmax
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
    dense2 = Dense(units=hp.Int('units', min_value=64, max_value=124, step=16), activation=activate_relu)(dense1)
    dropout_concat = (Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)))(dense2)
    finally_dense = (Dense(num_classes, activation='softmax'))(dropout_concat)
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
                         max_trials=10,
                         executions_per_trial=1,  # 각 모델별 학습 회수
                         directory='M',  # 사용된 parameter 저장할 폴더
                         project_name='mnist_tuner')  # 사용된 parameter 저장할 폴더

    # 튜너 학습
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True)
    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[callback], batch_size=batch_size)

    # 최적의 학습 모델 출력
    models = tuner.get_best_models(num_models=1)[0]
    print(models)
    
    # 모델 학습률 저장 
    models.save('cnn_tuner.h5')

    # 학습 결과 출력
    tuner.results_summary()
    score, acc = lstm_cnn_modeling().evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

    # prediction model
    prediction_result = lstm_cnn_modeling().predict(X_test)
    prediction_labels = np.argmax(prediction_result, axis=-1)
    test_label = np.argmax(y_test, axis=-1)
    
    
    # 건들지 마세요 model image classification test
    def prediction_data():
        wrong_result = []
        for n in range(0, len(test_label)):
            if prediction_labels[n] == test_label[n]:
                wrong_result.append(n)

        # 16개 임의로 선택
        sample = random.choices(population=wrong_result, k=25)
        count = 0
        nrows = ncols = 5
        plt.figure(figsize=(20, 8))
        for n in sample:
            count += 1
            plt.subplot(nrows, ncols, count)
            plt.imshow(X_test[n].reshape(28, 28), cmap="Greys", interpolation="nearest")
            tmp = "Label:" + str(test_label[n]) + ", Prediction:" + str(prediction_labels[n])
            plt.title(tmp)
        plt.tight_layout()
        plt.show()

    # confusion matrix making text file
    def print_matrix():
        confusion = print_confusion_matrix_v2(prediction_result, y_test)
        logw(file, f'Model Test loss -> {score} , Model Test accuracy -> {acc}')
        logw(file, f'Confusion Matrix -> \n' + np.array2string(confusion))
    
    
    print_matrix()
    prediction_data()
    


main()
