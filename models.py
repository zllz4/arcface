import math
import json
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

class NorminalizedDense(layers.Layer):
    def __init__(self, units, name="norminalized_dense", **kwargs):
        super(NorminalizedDense, self).__init__(name=name, **kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer=tf.random_normal_initializer(), trainable=True)
        self.s = self.add_weight(shape=(1,), initializer=tf.constant_initializer(value=1), trainable=True)
        # self.b = self.add_weight(shape=(self.units), initializer=tf.zeros_initializer(), trainable=True)
    def call(self, inputs):
        norm_inputs = tf.math.l2_normalize(inputs, axis=1)
        norm_w = tf.math.l2_normalize(self.w, axis=0)
        return tf.matmul(norm_inputs, norm_w) * self.s
    def get_config(self):
        config = {"units":self.units}
        base_config = super(NorminalizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MixFace(layers.Layer):
    def __init__(self, units, m1, m2, m3, name="mix_face", **kwargs):
        super(MixFace, self).__init__(name=name, **kwargs)
        self.units = units
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
    def build(self, input_shape):
        # input_shape: [TensorShape([None, 2]), TensorShape([None, 1])]
        self.w = self.add_weight(shape=(input_shape[0][-1], self.units), initializer='glorot_uniform', regularizer=tf.keras.regularizers.l2(1e-4), trainable=True)
        # self.s = self.add_weight(shape=(1,), initializer=tf.constant_initializer(value=30), trainable=False)
        self.s = 10
        # self.b = self.add_weight(shape=(self.units), initializer=tf.zeros_initializer(), trainable=True)

    def call(self, inputs, training=None):        
        x, y = inputs

        y = tf.reshape(y, (-1,))

        norm_inputs = tf.math.l2_normalize(x, axis=1)
        norm_w = tf.math.l2_normalize(self.w, axis=0)
        # fc: N x 10
        fc = tf.matmul(norm_inputs, norm_w)

        # tf.range() 默认出来的是 tf.int32
        # reshape 之后的 y 是 tf.float32
        origin_target_logit = tf.gather_nd(fc, tf.stack([tf.range(tf.shape(y)[0]), tf.cast(y,tf.int32)], axis=1))
        
        # tf.print(tf.math.acos(1.1)) -> nan, 因此需要 clip 操作,如果 σ 太小 (比如 1e-10 还是会造成 loss 变 nan)
        theta = tf.math.acos(tf.clip_by_value(origin_target_logit, -1+1e-6, 1-1e-6) )
       
        # 注意这步是使正确的选项的 scores 变小, 因此训练时 train 的 acc 有时会小于 10%
        # theta = theta * self.m1 + self.m2
        theta = tf.clip_by_value(theta * self.m1 + self.m2, 0, math.pi)
        marginal_target_logit = (tf.math.cos(theta) - self.m3)
        
        # N, -> N x 1 -> N x 10
        add_matrix = tf.reshape((marginal_target_logit-origin_target_logit), (-1,1)) * tf.one_hot(tf.cast(y,tf.int32), depth=tf.shape(norm_w)[1])
        
        # tf.print(x[1:5], summarize=-1)
        # tf.print(self.w,summarize=-1)
        # tf.print(fc[1:5],summarize=-1)
        # tf.print(theta[1:5],summarize=-1)
        # tf.print(y[1:5], summarize=-1)
        # tf.print(tf.argmax((fc + add_matrix), 1), summarize=-1)
        return (fc + add_matrix) * self.s
    
    def get_config(self):
        config = {"units":self.units, "m1":self.m1, "m2":self.m2, "m3":self.m3}
        base_config = super(MixFace, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_model(test_layer, add_input=False):
    weight_decay = 1e-4
    
    # for cifar-10
    # model_input = tf.keras.layers.Input((32,32,3))
    # x = model_input
    
    # for mnist
    model_input = tf.keras.layers.Input((28,28))
    x = tf.keras.layers.Reshape((28,28,1))(model_input)

    x = tf.keras.layers.Conv2D(16, 3, 1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = tf.keras.layers.Conv2D(32, 3, 1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = tf.keras.layers.Conv2D(64, 3, 1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(1000, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(2, name="2d_feature", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    
    if add_input:
        y = tf.keras.layers.Input((1,))
        x = test_layer([x, y])
    else:
        x = test_layer(x)
    
    model_output = tf.keras.layers.Activation('softmax')(x)
    if add_input:
        model = tf.keras.models.Model([model_input,y], model_output)
    else:
        model = tf.keras.models.Model(model_input, model_output)

    model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    print("----Layer Name In Full Model----")
    with open("model.json", "w") as f:
        json.dump(json.loads(model.to_json()), f, indent=1)
    for layer in json.loads(model.to_json())["config"]["layers"]:
        print("-",layer["config"]["name"])
    return model