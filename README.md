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

Arcface m=0.5

![arcface m=0.5](gif/Arcface%20m=0.5.gif)

Cosface m=0.1

![cosface m=0.1](gif/Cosface%20m=0.1.gif)

Sphereface m=1.35

![sphereface m=1.35](gif/Sphereface%20m=1.35.gif)

# 其它

目前在 cifar10 收敛上还有一些问题，可能模型某些设置不对

# 参考

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
