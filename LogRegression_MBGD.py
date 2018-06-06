# James Schnebly
# Chapter 9 Question 12
# Introduction to TensorFlow
# Logistic Regression w/ Mini-Batch Gradient Descent

import tensorflow as tf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

#reset default graph
tf.reset_default_graph

# Preprocessing
records = 1000
X_moons, y_moons = make_moons(records, noise = 0.1, random_state= 42)
X_moons_bias = np.c_[np.ones((records,1)), X_moons]
y_moons_column_vec = y_moons.reshape(-1,1)

# Vizualize the raw dataset
'''
plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label = "Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label = "Negative")
plt.legend()
plt.show()
'''

# Train Test Split
test_ratio = 0.2
test_size = int(records * test_ratio)
X_train = X_moons_bias[:-test_size]
X_test = X_moons_bias[-test_size:]
y_train = y_moons_column_vec[:-test_size]
y_test = y_moons_column_vec[-test_size:]

def random_batch(X_train, y_train, batch_size):
    rnd_indicies = np.random.randint(0 ,len(X_train), batch_size)
    X_batch = X_train[rnd_indicies]
    y_batch = y_train[rnd_indicies]
    return X_batch, y_batch

# Build the model
n_inputs = 2
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name = 'X') # input
y = tf.placeholder(tf.float32, shape=(None, 1), name = 'y') # output
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name = 'theta') #weights
logits = tf.matmul(X, theta, name='logits') # operation (X * theta)
y_proba = tf.sigmoid(logits) # appy sigmoid to logits
loss = tf.losses.log_loss(y, y_proba) # loss function
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # create optimizer (Gradient Descent)
training_operation = optimizer.minimize(loss) # apply optimizer to loss function
init = tf.global_variables_initializer()

# run the model
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(records/batch_size))

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_operation, feed_dict={X: X_batch, y: y_batch})
        
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
    
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    
y_pred = (y_proba_val >= .5)

# Evaluate the model
from sklearn.metrics import precision_score, recall_score
accuracy = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision:", accuracy)
print("Recall:", recall)







