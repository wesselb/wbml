import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab.tf import B

from wbml import feedforward

x = np.linspace(-10, 10, 100, dtype=np.float32)[None, :]
y = x ** 2

y_nn = feedforward([1, 10, 10, 1])(x)
obj = B.mean((y - y_nn) ** 2) ** .5

s = tf.Session()
opt = tf.train.AdamOptimizer(5e-2).minimize(obj)
s.run(tf.global_variables_initializer())

for i in range(2000):
    _, val = s.run([opt, obj])
    if i % 100 == 0:
        print(i, val)

x, y = x.flatten(), y.flatten()
y_learned = s.run(y_nn).flatten()
plt.figure()
plt.plot(x, y.flatten(), label='True', marker='o')
plt.plot(x, y_learned, label='Learned', marker='x')
plt.legend()
plt.show()
