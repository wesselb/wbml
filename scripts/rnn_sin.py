import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab.tf import B

from wbml.data import normalise_01, normalise_norm
from wbml import rnn

x = np.linspace(-20, 20, 80, dtype=np.float32)[:, None, None]
y = np.sin(x)

# Normalise inputs and outputs.
x = normalise_01(x)
y = normalise_norm(y)

# Split data.
i = int(x.shape[0] * .5)
x_train = x[:i, :, :]
y_train = y[:i, :, :]

f = rnn(1, 1, ((10, 10),), gru=True)
y_rnn = f(x_train)
obj = B.mean((y_train - y_rnn) ** 2)

s = tf.Session()
opt = tf.train.AdamOptimizer(1e-2).minimize(obj)
s.run(tf.global_variables_initializer())

for i in range(3000):
    _, val = s.run([opt, obj])
    if i % 100 == 0:
        print(i, val)

y_learned = s.run(f(x)).flatten()
x, y = x.flatten(), y.flatten()
x_train, y_train = x_train.flatten(), y_train.flatten()
plt.figure()
plt.plot(x, y, label='True', ls='-', marker='o')
plt.plot(x_train, y_train, label='Observed', marker='o', ls='none')
plt.plot(x, y_learned, label='Learned', marker='x')
plt.legend()
plt.show()
