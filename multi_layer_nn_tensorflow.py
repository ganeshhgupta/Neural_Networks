# Setu, Mehzad Hossain
# 1002_138_330
# 2023_10_14
# Assignment_02_01


import numpy as np
import tensorflow as tf

def mse_loss(y_true, y_pred):
    loss = tf.reduce_mean((y_true - y_pred) ** 2)
    return tf.reduce_mean(loss)
def svm_loss(y_true, y_pred):
    margin = 1 - y_true * y_pred
    loss = tf.maximum(0.0, margin)
    return tf.reduce_mean(loss)

def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="mse",
                              validation_split=[0.8, 1.0], seed=2):
    # Split the data into training and validation sets
    num_samples = X_train.shape[0]
    start_idx = int(validation_split[0] * num_samples)
    end_idx = int(validation_split[1] * num_samples)
    X_val, Y_val = X_train[start_idx:end_idx], Y_train[start_idx:end_idx]
    if validation_split[1]==1:
        X_train, Y_train = X_train[:start_idx], Y_train[:start_idx]
    else:
        X_train = np.concatenate((X_train[:start_idx], X_train[end_idx:]), axis=0)
        Y_train = np.concatenate((Y_train[:start_idx], Y_train[end_idx:]), axis=0)
    # Initialize weights
    if isinstance(layers[0], int):
        input_size = X_train.shape[1]
        layers = [input_size] + layers
        weights = []
        for i in range(1, len(layers)):
            np.random.seed(seed)
            weight_layer = tf.Variable(np.random.randn(layers[i - 1] + 1, layers[i]).astype(np.float32))
            weights.append(weight_layer)
    else:
        weights = layers

    # Activation functions
    activation_functions = {
        "linear": tf.identity,
        "sigmoid": tf.sigmoid,
        "relu": tf.nn.relu
    }

    error_history = []

    # Initialize activations_val
    activations_val = [X_val]
    for _ in range(len(layers) - 1):
        activations_val.append(None)

    # Determine the loss function
    if loss.lower() == "mse":
        loss_function = mse_loss
    elif loss.lower() == "svm":
        loss_function = svm_loss
    elif loss.lower() == "cross_entropy":
        loss_function = cross_entropy_loss
    else:
        raise ValueError("Unsupported loss function: {}".format(loss))

    for j in range(len(layers) - 1):
        activation_fn = activation_functions[activations[j]]
        Z = tf.matmul(tf.concat([tf.ones((X_val.shape[0], 1)), activations_val[j]], axis=1), weights[j])
        A = activation_fn(Z)
        activations_val[j + 1] = A

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                # Forward pass
                activations_list = [X_batch]
                for j in range(len(layers) - 1):
                    activation_fn = activation_functions[activations[j]]
                    Z = tf.matmul(tf.concat([tf.ones((X_batch.shape[0], 1)), activations_list[j]], axis=1), weights[j])
                    A = activation_fn(Z)
                    activations_list.append(A)

                # Calculate loss
                loss = loss_function(Y_batch, activations_list[-1])

            # Compute gradients
            grads = tape.gradient(loss, weights)

            # Update weights
            for j in range(len(layers) - 1):
                weights[j].assign_sub(alpha * grads[j])

        # Calculate validation loss
        for j in range(len(layers) - 1):
            activation_fn = activation_functions[activations[j]]
            Z = tf.matmul(tf.concat([tf.ones((X_val.shape[0], 1)), activations_val[j]], axis=1), weights[j])
            A = activation_fn(Z)
            activations_val[j + 1] = A

        val_loss = loss_function(Y_val, activations_val[-1])
        error_history.append(val_loss.numpy())

    return weights, error_history, np.array(activations_val[-1])
