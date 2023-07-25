import numpy
import tensorflow as tf

learning_rate = 1e-3


def update_weights(gradients, weights):
  for g, w in zip(gradients, weights):
    w.assign_sub(g * learning_rate)  # ←---- assign_sub相当于TensorFlow变量的-=


# from tensorflow.keras import optimizers
#
# optimizer = optimizers.SGD(learning_rate=1e-3)
#
# def update_weights(gradients, weights):
#   optimizer.apply_gradients(zip(gradients, weights))


class NaiveDense:
  def __init__(self, input_size, output_size, activation):
    self.activation = activation

    w_shape = (input_size, output_size)  # ←----创建一个形状为(input_size, output_size)的矩阵W，并将其随机初始化
    w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
    self.W = tf.Variable(w_initial_value)

    b_shape = (output_size,)  # ←----创建一个形状为(output_size,)的零向量b
    b_initial_value = tf.zeros(b_shape)
    self.b = tf.Variable(b_initial_value)

  def __call__(self, inputs):  # ←----前向传播
    return self.activation(tf.matmul(inputs, self.W) + self.b)

  @property
  def weights(self):  # ←----获取该层权重的便捷方法
    return [self.W, self.b]


class NaiveSequential:
  def __init__(self, layers):
    self.layers = layers

  def __call__(self, inputs):
    x = inputs
    for layer in self.layers:
      x = layer(x)
    return x

  @property
  def weights(self):
    weights = []
    for layer in self.layers:
      weights += layer.weights
    return weights


model = NaiveSequential([
  NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
  NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4

import math


class BatchGenerator:
  def __init__(self, images, labels, batch_size=128):
    assert len(images) == len(labels)
    self.index = 0
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.num_batches = math.ceil(len(images) / batch_size)

  def next(self):
    images = self.images[self.index: self.index + self.batch_size]
    labels = self.labels[self.index: self.index + self.batch_size]
    self.index += self.batch_size
    return images, labels


def one_training_step(model, images_batch, labels_batch):
  with tf.GradientTape() as tape:  # ←---- (本行及以下4行) 运行前向传播，即在GradientTape作用域内计算模型预测值
    predictions = model(images_batch)
    per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
      labels_batch, predictions)
    average_loss = tf.reduce_mean(per_sample_losses)

    gradients = tape.gradient(average_loss,
                              model.weights)  # ←----计算损失相对于权重的梯度。输出gradients是一个列表，每个元素对应model.weights列表中的权重
    update_weights(gradients, model.weights)  # ←----利用梯度来更新权重（稍后给出这个函数的定义）
    return average_loss


def fit(model, images, labels, epochs, batch_size=128):
  for epoch_counter in range(epochs):
    print(f"Epoch {epoch_counter}")
    batch_generator = BatchGenerator(images, labels)
    for batch_counter in range(batch_generator.num_batches):
      images_batch, labels_batch = batch_generator.next()
      loss = one_training_step(model, images_batch, labels_batch)
      if batch_counter % 100 == 0:
        print(f"loss at batch {batch_counter}: {loss:.2f}")


from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=20, batch_size=128)

predictions = model(test_images)
predictions = predictions.numpy()  # ←----对TensorFlow张量调用.numpy()，可将其转换为NumPy张量
predicted_labels = numpy.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
