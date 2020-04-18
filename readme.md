# 环境安装

```shell
conda create -n arcface
conda install tensorflow-gpu=2.1 // tensorflow-gpu 2.0 版本的 model.evaluate() 函数有问题
conda install matplotlib
```

# 运行

```shell
python arcface.py
```

取消注释使用其它损失

```python
# Softmax
# model = models.create_model(tf.keras.layers.Dense(10, name="test_layer"))

# Norm Softmax
# model = models.create_model(models.NorminalizedDense(10, name="test_layer"))

# Arcface
# model = models.create_model(models.MixFace(10, m1=1, m2=0.5, m3=0, name="test_layer"), add_input=True)
# ADD_INPUT = True

# Sphereface
# model = models.create_model(models.MixFace(10, m1=1.35, m2=0, m3=0, name="test_layer"), add_input=True)
# ADD_INPUT = True

# Cosface
# m3=0.35 mnist 数据集也无法收敛
model = models.create_model(models.MixFace(10, m1=1, m2=0, m3=0.1, name="test_layer"), add_input=True)
ADD_INPUT = True
```

# 结果

普通 softmax

![softmax](gif/Softmax.gif)

归一化 softmax

![norm softmax](gif/Norm%20Softmax.gif)

Arcface m=0.1

![arcface m=0.1](gif/Arcface%20m=0.1.gif)

Arcface m=0.5（用的是 mnist 数据集，cifar 10 收敛不了）

![arcface m=0.5](gif/Arcface%20m=0.5.gif)

Cosface m=0.1

![cosface m=0.1](gif/Cosface%20m=0.1.gif)

Sphereface m=1.35

![sphereface m=1.35](gif/Sphereface%20m=1.35.gif)

# 其它

由于换成 arcface 等基于 angular margin 的损失之后很难训练，很多时候一直都不收敛（好不容易换成 SGD 优化器突然能收敛了，结果一看 θ 全是 3.6，由于 θ + m 没有上限限制，全都超过 pi 了。。。）本来想一直用 cifar-10 测试的但是最后只能换成 mnist，但还是有些参数即使 mnist 也无法成功收敛，可能是模型设置还有哪里有问题