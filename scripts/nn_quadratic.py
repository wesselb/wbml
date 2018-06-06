import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab.tf import B

from wbml import ff, normalise_01, normalise_norm

x = np.linspace(-10, 10, 100, dtype=np.float32)[None, :]
y = x ** 2

# Normalise inputs and outputs.
x = normalise_01(x)
y = normalise_norm(y)

y_nn = ff(1, 1, (10, 10))(x)
obj = B.mean((y - y_nn) ** 2)

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
plt.plot(x, y, label='True', marker='o')
plt.plot(x, y_learned, label='Learned', marker='x')
plt.legend()
plt.show()
