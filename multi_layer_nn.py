# Setu, Mehzad Hossain
# 1002_138_330
# 2023_09_24
# Assignment_01_01


import numpy as np

def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))


def mse(y_true, y_pred):
    # Mean Squared Error (MSE) loss function
    return np.mean((y_true - y_pred) ** 2)

def fwd_prop(weights, layer_outputs, layer_input):
    for w in weights:
        layer_input = np.dot(w, layer_outputs)
        layer_output = sigmoid(layer_input)
        value_to_insert = 1
        new_array = np.array([value_to_insert])
        bla = np.concatenate((new_array, layer_output))
        layer_outputs = bla

def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):
    epsilon = h  # Small value for finite difference
    out = layers[-1]

    X_train = X_train.T
    X_test = X_test.T
    input_size = X_train.shape[1]

    ones_row = np.ones((X_train.shape[0], 1))
    result = np.hstack((ones_row, X_train))

    ones_row = np.ones((X_test.shape[0], 1))
    result_test = np.hstack((ones_row, X_test))

    layer_sizes = [input_size] + layers

    # Initialize weight
    weights = []
    for i in range(1, len(layer_sizes)):
        np.random.seed(seed)
        weight_layer = np.random.randn(layer_sizes[i], layer_sizes[i - 1] + 1)
        weights.append(weight_layer)

    final_mse = []

    output_test = []
    for a in range(len(X_test)):
        layer_outputs = result_test[a]

        for i in range(len(weights)):
            layer_input = np.dot(weights[i], layer_outputs)
            #print(layer_input)
            layer_output = sigmoid(layer_input)
            value_to_insert = 1
            new_array = np.array([value_to_insert])
            bla = np.concatenate((new_array, layer_output))

            layer_outputs = bla
        output_test.append(bla[-out:])

    final_output = np.array(output_test).T
    #     final_mse.append(mse(Y_test, final_output))

    # Training
    for epoch in range(epochs):
        output = []

        for a in range(len(X_train)):
            layer_outputs = result[a]

            # Store weight updates for each layer
            weight_updates = [np.zeros_like(w) for w in weights]

            for i in range(len(weights)):
                layer_input = np.dot(weights[i], layer_outputs)
                layer_output = sigmoid(layer_input)
                value_to_insert = 1
                new_array = np.array([value_to_insert])
                bla = np.concatenate((new_array, layer_output))
                layer_outputs = bla

            output.append(bla[-out:])
            output_train = np.array(output).T

            # Compute gradients using finite differences
            for i in range(len(weights)):
                for j in range(weights[i].shape[0]):
                    for k in range(weights[i].shape[1]):
                        original_value = weights[i][j, k]

                        # Perturb the weight slightly
                        weights[i][j, k] = original_value + epsilon

                        # Forward pass with perturbed weight
                        layer_outputs = result[a]
                        for w in weights:
                            layer_input = np.dot(w, layer_outputs)
                            layer_output = sigmoid(layer_input)
                            value_to_insert = 1
                            new_array = np.array([value_to_insert])
                            bla = np.concatenate((new_array, layer_output))
                            layer_outputs = bla

                        output_epsilon_p = bla[-out:]
                        plus_error = mse(Y_train[:, a], output_epsilon_p)

                        ## Perturb the weight slightly negative
                        weights[i][j, k] = original_value - epsilon

                        # Forward pass with negatively perturbed weight
                        layer_outputs = result[a]
                        for w in weights:
                            layer_input = np.dot(w, layer_outputs)
                            layer_output = sigmoid(layer_input)
                            value_to_insert = 1
                            new_array = np.array([value_to_insert])
                            bla = np.concatenate((new_array, layer_output))
                            layer_outputs = bla

                        output_epsilon_m = bla[-out:]
                        minus_error = mse(Y_train[:, a], output_epsilon_m)

                        # Restore the original weight value
                        weights[i][j, k] = original_value

                        # Calculate the gradient using centered difference approximation
                        gradient = (plus_error - minus_error) / (2 * epsilon)
                        weight_updates[i][j, k] = -alpha * gradient  # Negative gradient for weight update

            # Update weights for this sample using accumulated weight updates
            for i in range(len(weights)):
                weights[i] += weight_updates[i]

        # Calculate MSE for test data after each epoch
        output_test = []
        for a in range(len(X_test)):
            layer_outputs = result_test[a]

            for i in range(len(weights)):
                layer_input = np.dot(weights[i], layer_outputs)
                layer_output = sigmoid(layer_input)
                value_to_insert = 1
                new_array = np.array([value_to_insert])
                bla = np.concatenate((new_array, layer_output))

                layer_outputs = bla
            output_test.append(bla[-out:])

        final_output = np.array(output_test).T
        final_mse.append(mse(Y_test, final_output))

        #print(f'Epoch {epoch + 1}, MSE: {mse(Y_train, output_train)}')
    return [weights, final_mse, final_output]
