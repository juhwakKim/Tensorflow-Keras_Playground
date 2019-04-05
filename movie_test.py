# https://www.tensorflow.org/tutorials/keras/basic_text_classification
import tensorflow as tf
from tensorflow import keras

import numpy as np

load_model = True

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,))) #https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                loss= keras.losses.binary_crossentropy,
                metrics=['acc'])

    return model
            

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

word_index = imdb.get_word_index()
# 단어와 정수 인덱스를 매핑한 딕셔너리
word_index = {k:(v+3) for k,v in word_index.items()}
# 처음 몇 개 인덱스는 사전에 정의되어 있습니다
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text): 
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post' ,
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post' ,
                                                        maxlen=256)

vocab_size = 10000

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

if(load_model == False):

    model = create_model()

    tb_hist = keras.callbacks.TensorBoard(log_dir='./model/movie_text/graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True) #텐서보드 연동 https://tykimos.github.io/2017/07/09/Training_Monitoring/

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[tb_hist]) #verbose=:logging 0:없음 1:프로그레스바, 2:epoch당

    results = model.evaluate(test_data, test_labels)
    print(results)

    model.save('./model/movie_text/movie_test_model.h5') #폴더가 실제 있어야함

if(load_model == True):
    new_model = keras.models.load_model('./model/movie_text/movie_test_model.h5') 
    new_model.summary()
    loss, acc = new_model.evaluate(test_data, test_labels) #optimizer를 tf.train.AdamOptimizer() 대신 'adam' or keras.optimizers.Adam() 으로 작성해야 작동(아마 loss도 keras이외 모듈에서 가져오면 오류가 생길것으로 보임)
    print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))





