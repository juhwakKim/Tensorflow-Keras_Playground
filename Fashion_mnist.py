# https://www.tensorflow.org/tutorials/keras/basic_classification
# 세이브 https://www.tensorflow.org/tutorials/keras/save_and_restore_models
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

checkpoint_path = "./Keras_MNIST/cp-{epoch:04d}ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
load_model_weight = False
load_model = True

def create_model():
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation= tf.nn.relu),
    keras.layers.Dense(10, activation= tf.nn.softmax)])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 데이터 전처리
train_images = train_images / 255.0

test_images = test_images / 255.0

if(load_model_weight == False and load_model == False):


    #estimator_model = tf.keras.estimator.model_to_estimator(keras_model = model, \
    #                                                       model_dir = './Keras_MNIST') #ckpt형태로 저장 https://towardsdatascience.com/freezing-a-keras-model-c2e26cb84a38

    model =create_model()

    # 체크포인트 콜백 만들기 
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1)

    tb_hist = keras.callbacks.TensorBoard(log_dir='./model/Fashion_mnist/graph', histogram_freq=0, write_graph=True, write_images=True, write_grads=True) #텐서보드 연동 https://tykimos.github.io/2017/07/09/Training_Monitoring/

    model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback,tb_hist])

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('테스트 정확도:', test_acc)

    predictions = model.predict(test_images)

    print("예측",np.argmax(predictions[0]))

    print("정답",test_labels[0])

    model.save('./model/Fashion_mnist/mnist_mlp_model.h5') #폴더가 실제 있어야함


if(load_model_weight == True):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = create_model()
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels)
    print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

if(load_model == True):
    new_model = keras.models.load_model('./model/Fashion_mnist/mnist_mlp_model.h5')
    print(new_model.layers[1].get_weights()[0].name)
    new_model.summary()
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

