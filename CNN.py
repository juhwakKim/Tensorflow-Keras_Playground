# Estimator: https://bcho.tistory.com/1196
# korean: https://reniew.github.io/03/
from __future__ import absolute_import, division, print_function #http://www.hanbit.co.kr/channel/category/category_view.html?cms_code=CMS9324226566

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO) #함수 실행시 Info print해줌


def cnn_mode_fn(features, labels, mode):
    
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.Conv2D(
        inputs= input_layer,
        filters= 32,
        kernel_size= [5, 5],
        padding= "same",
        activation= tf.nn.relu,
        name= "conv1"
        )
        
    pool1 = tf.layers.max_pooling2d(inputs= conv1, pool_size= [2, 2], strides=2, name= "pool1")

    conv2 = tf.layers.Conv2D(
        inputs= pool1,
        filters= 64,
        kernel_size= [5, 5],
        padding= "same",
        activation= tf.nn.relu,
        name= "conv2"
        )
        
    pool2 =tf.layers.max_pooling2d(inputs= conv2, pool_size= [2, 2], strides= 2, name= "pool2")

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs= pool2_flat, units= 1024, activation= tf.nn.relu, name= "dense1")
    dropout = tf.layers.dropout(
        inputs= dense, rate= 0.4, training= mode == tf.estimator.ModeKeys.TRAIN, name= "dropout")

    logits = tf.layers.dense(inputs= dropout, units= 10, name= "dense2")

    predictions = {
        "classes": tf.argmax(input= logits, axis= 1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode= mode, predictions= predictions) # Estimator의 리턴값 EstimatorSpec객체

    loss = tf.losses.sparse_softmax_cross_entropy(labels= labels, logits= logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.001)
        train_op = optimizer.minimize(loss= loss, global_step= tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode= mode, loss= loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode= mode, loss= loss, eval_metric_ops= eval_metric_ops)

#######
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
####### cnn algorithm error
        
((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)  

eval_data = eval_data/np.float32(255)
eval_labels = eval_labels.astype(np.int32)  

mnist_classifier = tf.estimator.Estimator(
    model_fn= cnn_mode_fn, model_dir="./model/mnist_convnet_model") #model_dir 없으면 저장 x

tensors_to_log = {"probabilities": "softmax_tensor"} # INFO로 name의 내요을 print해줌

logging_hook = tf.train.LoggingTensorHook(           # logging할 방법을 정한다.
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn  =tf.estimator.inputs.numpy_input_fn(
    x= {"x": train_data},
    y= train_labels,
    batch_size= 100,
    num_epochs= None,
    shuffle= True)

mnist_classifier.train(
    input_fn= train_input_fn,
    steps= 1,
    hooks= [logging_hook])

train_ = mnist_classifier.train(input_fn=train_input_fn, steps=1000)#

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x= {"x": eval_data},
    y= eval_labels,
    num_epochs= 1,
    shuffle= False)

eval_results = mnist_classifier.evaluate(input_fn= eval_input_fn)
print(eval_results)




        

        


