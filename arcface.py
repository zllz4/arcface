import os
import json
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist

import models

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(tf.shape(x_train))
print(tf.shape(y_train))

ADD_INPUT = False

# Softmax
# model = models.create_model(tf.keras.layers.Dense(10, name="test_layer"))

# Norm Softmax
model = models.create_model(models.NorminalizedDense(10, name="test_layer"))

# Arcface
# model = models.create_model(models.MixFace(10, m1=1, m2=0.5, m3=0, name="test_layer"), add_input=True)
# ADD_INPUT = True

# Sphereface
# model = models.create_model(models.MixFace(10, m1=1.35, m2=0, m3=0, name="test_layer"), add_input=True)
# ADD_INPUT = True

# # Cosface
# # m3=0.35 mnist 数据集也无法收敛
# model = models.create_model(models.MixFace(10, m1=1, m2=0, m3=0.1, name="test_layer"), add_input=True)
# ADD_INPUT = True


fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_ylim()
ax1.set_ylim(-1.2,1.2)
ax1.set_xlim(-1.2,1.2)
ax2 = fig.add_subplot(1,2,2)
ax2.set_ylim(0, 1)

def get_rand_index(num, dataset="train"):
    if dataset == "train":
        return [random.randint(0,len(x_train)-1) for i in range(num)]
    else:
        return [random.randint(0,len(x_test)-1) for i in range(num)]

def get_2d_feature():
    epoch = 0
    acc_list = [0.0]
    rand_1000_index = get_rand_index(1000, "test")
    while epoch < 100:
        print("epoch:",epoch)
        epoch += 1
        if ADD_INPUT:
            history = model.fit([x_train, y_train], y_train, epochs=1, batch_size=128)
            loss, acc = model.evaluate([x_test, y_test], y_test, batch_size=128, verbose=1)
            model_2d_feature = tf.keras.Model(inputs=model.input, outputs=model.get_layer("2d_feature").output)
            feature = model_2d_feature.predict([x_test[rand_1000_index],y_test[rand_1000_index]])
            # loss = history.history['loss']
            # acc = history.history['sparse_categorical_accuracy'][0]
        else:
            history = model.fit(x_train, y_train, epochs=1)
            loss, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
            model_2d_feature = tf.keras.Model(inputs=model.input, outputs=model.get_layer("2d_feature").output)
            feature = model_2d_feature.predict(x_test[rand_1000_index])
            # loss = history.history['loss']
            # acc = history.history['sparse_categorical_accuracy']

        feature = tf.math.l2_normalize(feature, 1)
        label = tf.reshape(y_test[rand_1000_index], (-1,))
        acc_list.extend([acc])
        weight = model.get_layer("test_layer").get_weights()
        print(weight)
        yield (feature, label, range(epoch+1), acc_list, weight[0]/np.linalg.norm(weight[0], axis=0).reshape((1,-1)))

def show_result(result):
    ax1.cla()
    ax2.cla()
    ax1.set_title("Cosface m=0.1 2D Feature (mnist)")
    ax2.set_title("Accuracy")
    feature, label, epoch, acc, weight = result
    ax1.scatter(feature[:, 0], feature[:, 1], c=[plt.cm.tab10(i/9) for i in label], cmap=plt.get_cmap('tab10'), s=1)
    # plt.hold(True)
    for i in range(10):
        ax1.plot([0, weight[0,i]], [0, weight[1,i]], c=plt.cm.tab10(i/9))
    # print(epoch, acc)
    ax2.plot(epoch, acc)
    # plt.hold(False)
    
feature_generator = get_2d_feature()
    

ani = FuncAnimation(fig, show_result, frames=feature_generator, repeat=False)
plt.show()
# ani.save("2d_feature.gif", writer='imagemagick')



