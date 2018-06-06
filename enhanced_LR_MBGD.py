# -*- coding: utf-8 -*-
# James Schnebly
# Chapter 9 Question 12
# Introduction to TensorFlow
# Logistic Regression w/ Mini-Batch Gradient Descent (Enhanced)

import tensorflow as tf
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

#reset default graph
tf.reset_default_graph

# Preprocessing
records = 1000
X_moons, y_moons = make_moons(records, noise = 0.1, random_state= 42)
X_moons_bias = np.c_[np.ones((records,1)), X_moons]
y_moons_column_vec = y_moons.reshape(-1,1)

# Train Test Split
test_ratio = 0.2
test_size = int(records * test_ratio)
X_train = X_moons_bias[:-test_size]
X_test = X_moons_bias[-test_size:]
y_train = y_moons_column_vec[:-test_size]
y_test = y_moons_column_vec[-test_size:]

# Enhance the dataset
X_train_enhanced = np.c_[X_train,                  # input data X
                         np.square(X_train[:, 1]), # X_1 squared
                         np.square(X_train[:, 2]), # X_2 squared
                         X_train[:, 1] ** 3,       # X_1 cubed
                         X_train[:, 2] ** 3]       # X_2 cubed

X_test_enhanced = np.c_[X_test,                   # input data X
                         np.square(X_test[:, 1]), # X_1 squared
                         np.square(X_test[:, 2]), # X_2 squared
                         X_test[:, 1] ** 3,       # X_1 cubed
                         X_test[:, 2] ** 3]       # X_2 cubed

# random batch function
def random_batch(X_train, y_train, batch_size):
    rnd_indicies = np.random.randint(0 ,len(X_train), batch_size)
    X_batch = X_train[rnd_indicies]
    y_batch = y_train[rnd_indicies]
    return X_batch, y_batch


# create logistic regression model
def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
            
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_operation = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
            
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
            
        with tf.name_scope("save"):
            saver = tf.train.Saver()
            
    return y_proba, loss, training_operation, loss_summary, init, saver
            
# log function
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

# prepare to run
n_inputs = 2 + 4
logdir = log_dir("logreg")

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name = 'X')
y = tf.placeholder(tf.float32, shape=(None, 1), name = 'y')            

y_proba, loss, training_operation, loss_summary, init, saver = logistic_regression(X, y)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())            
 
# run
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(records / batch_size))

checkpoint_path = "tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model"

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
        
    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
            sess.run(training_operation, feed_dict = {X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
            
        file_writer.add_summary(summary_str, epoch)
            
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
    
    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict = {X: X_test_enhanced, y: y_test})
    y_pred = (y_proba_val >= .5)
    os.remove(checkpoint_epoch_path)

# Evaluate the model
from sklearn.metrics import precision_score, recall_score
accuracy = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision:", accuracy)
print("Recall:", recall)
        
















            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            