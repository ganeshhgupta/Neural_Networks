# Setu, Mehzad Hossain
# 1002_138_330
# 2023_10_27
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, n_classes=10):
    # Initialize an empty confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4, validation_split=0.2):
    # Set a random seed for reproducibility
    tf.keras.utils.set_random_seed(5368)

    model = tf.keras.models.Sequential([
        # - Convolutional layer with 8 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
        tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(28, 28, 1)),

        # - Convolutional layer with 16 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
        tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

        # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # - Convolutional layer with 32 filters, kernel size 3 by 3, stride 1 by 1, padding 'same', and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

        # - Convolutional layer with 64 filters, kernel size 3 by 3 , stride 1 by 1, padding 'same', and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

        # - Max pooling layer with pool size 2 by 2 and stride 2 by 2
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # - Flatten layer
        tf.keras.layers.Flatten(),

        # - Dense layer with 512 units and ReLU activation
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

        # - Dense layer with 10 units with linear activation
        tf.keras.layers.Dense(10, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),

        # - a softmax layer
        tf.keras.layers.Activation('softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    _, accuracy = model.evaluate(X_test, Y_test)

    Y_pred = model.predict(X_test)

    Y_true_new = np.argmax(Y_test, axis=1)
    Y_pred_new = np.argmax(Y_pred, axis=1)
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for i in range(len(Y_true_new)):
        confusion_matrix[Y_true_new[i], Y_pred_new[i]] += 1

    # Plot the confusion matrix
    plt.matshow(confusion_matrix, cmap='Greens')
    plt.colorbar()
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')

    # Save the model
    model.save('model.h5')

    return [model, history, confusion_matrix, Y_pred_new]

